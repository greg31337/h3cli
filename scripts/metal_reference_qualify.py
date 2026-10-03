#!/usr/bin/env python3
"""Serial 243-frame M2 Reference qualification with explicit acceptance records.

Performance acceptance is separate. All six-step runs use the production VAE.
No human approval gate and no production-default promotion are performed here.
"""
import argparse
from array import array
import hashlib
import html
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
from metal_native_bench import ROOT, sha


def check_hashes(entries):
    for name,digest in entries.items():
        if sha(ROOT/name)!=digest:raise ValueError('Changed frozen evidence: '+name)


def quality_acceptance(path,binary,contract):
    a=json.loads(path.read_text());check_hashes(a['evidence_sha256'])
    if (a['version']!=1 or a['binary_sha256']!=sha(binary) or a['geometry']!=[640,480,243] or
        a['maximum_evaluations']!=6 or a['mixed_recipe']!=4 or
        a['evidence_sha256'].get('tests/metal_reference_contract.json')!=sha(contract)):
        raise ValueError('Quality acceptance does not apply to this executable/contract')
    index=ROOT/a['accepted_index']
    if a['evidence_sha256'].get(a['accepted_index'])!=sha(index):raise ValueError('Unbound accepted index')
    prior=json.loads(index.read_text())
    if prior['binary_sha256']!=a['binary_sha256']:raise ValueError('Accepted executable mismatch')
    row=next(r for r in prior['cases'] if r['id']=='calibration')
    for label in ('reference','candidate'):
        if sha(Path(row[label])/'record.json')!=row['record_sha256'][label]:raise ValueError('Changed accepted run')
    check_pair(Path(row['reference']),Path(row['candidate']))
    metric_path=index.parent/row['quality']/'metrics.json'
    metrics=json.loads(metric_path.read_text())
    if sha(metric_path)!=row['metrics_sha256'] or metrics['contract_sha256']!=sha(contract):
        raise ValueError('Changed accepted quality metrics')
    return a,index,row


def av_payload(path):
    b=path.read_bytes()
    if len(b)<160 or b[:8]!=b'H3AV\r\n\x1a\n':raise ValueError('Invalid AV state')
    nv,na=struct.unpack_from('<QQ',b,72)
    if len(b)!=160+nv+na or nv%4 or na%4:raise ValueError('Invalid AV lengths')
    if hashlib.sha256(b[:128]+b[160:]).digest()!=b[128:160]:raise ValueError('AV checksum mismatch')
    f=array('f');f.frombytes(b[160:])
    if sys.byteorder!='little':f.byteswap()
    if not all(math.isfinite(x) for x in f):raise ValueError('Nonfinite final AV state')
    return struct.unpack_from('<10I',b,24),(b[160:160+nv],b[160+nv:]),b[88:120]


def validate_run(path,native):
    r=json.loads((path/'record.json').read_text())
    if (r.get('teacher_from') or not r['evaluation_contract_pass'] or r['returncode'] or
        r['case']!='B6' or r['evaluations']!=6 or r['blocks']!=50 or r['preview_vae'] or
        r['geometry']!=[640,480,243] or r['weight_precision']!='bf16' or r['attention']!='dense' or
        r['sampler']!='cpu-euler' or r['backend']!=('metal' if native else 'mpsgraph')):
        raise ValueError('Incomplete/nonindependent B6 qualification: '+str(path))
    if r['av_sha256']!=sha(path/'result.h3av') or r['media_sha256']!=sha(path/'result.mp4'):
        raise ValueError('Changed output')
    if r['manifest']['binary_sha256']!=sha(path/'bin/h3cli'):raise ValueError('Changed binary')
    if len(r['steps'])!=6 or [x['step'] for x in r['steps']]!=list(range(1,7)):
        raise ValueError('Missing/repeated evaluation')
    if any(not x['evaluated'] or x['native_attention' if native else 'mps_attention']!=50 for x in r['steps']):
        raise ValueError('Missing full-block execution')
    if len(r['recipes'])!=6 or any(x['routing']!='dense' or x['state_dtype']!='bf16/fp32' or
        x['weight_storage']!='bf16-source' for x in r['recipes']):raise ValueError('Changed arithmetic recipe')
    if native:
        if any(x['attention_storage']!='fp16' or x['mixed_recipe']!=4 for x in r['recipes']):
            raise ValueError('Unexpected mixed precision recipe')
        rows=r['mixed']
        if (len(rows)!=300 or {(x['step'],x['block']) for x in rows}!={(s,b) for s in range(1,7) for b in range(50)} or
            any(x['heads']!=56 or x['invalid_heads'] for x in rows)):
            raise ValueError('Missing/invalid range handling')
    av_payload(path/'result.h3av')
    return r


def check_pair(reference,candidate,source=None):
    r=validate_run(reference,False);c=validate_run(candidate,True)
    for key in ('geometry','seed','prompt','weight_precision','sampler','conditioning_sha256_after',
                'conditioning_identity','reference_sha256','extra_inputs_sha256','continuation_sha256','continuation_context'):
        if r.get(key)!=c.get(key) and not (key=='extra_inputs_sha256' and not r.get(key) and not c.get(key)):
            raise ValueError('Unmatched input: '+key)
    rg,rp,rs=av_payload(reference/'result.h3av');cg,cp,cs=av_payload(candidate/'result.h3av')
    if rg!=cg or rs!=cs:raise ValueError('Final state geometry/signature mismatch')
    result={'finite':True,'independent':True,'full_evaluations':True,'prefix':None}
    context=r.get('continuation_context')
    if context:
        if source is None or sha(source)!=r['continuation_sha256']:raise ValueError('Missing/changed continuation source')
        sg,sp,ss=av_payload(source)
        if ss!=rs or sg[:2]!=rg[:2] or sg[4:6]!=rg[4:6]:raise ValueError('Incompatible source')
        prefix={};k=(context-39)//51
        for modality,i,frames,spatial,channels,count in (
            ('video',0,3,rg[4]*rg[5],24,12+15*k),('audio',1,6,1,64,65+85*k)):
            stride=rg[frames]*spatial*4;source_stride=sg[frames]*spatial*4;cut=count*spatial*4
            expected=b''.join(sp[i][(ch+1)*source_stride-cut:(ch+1)*source_stride] for ch in range(channels))
            for payload in (rp[i],cp[i]):
                actual=b''.join(payload[ch*stride:ch*stride+cut] for ch in range(channels))
                if actual!=expected:raise ValueError('Inherited '+modality+' prefix changed')
            prefix[modality]={'bytes':len(expected),'sha256':hashlib.sha256(expected).hexdigest(),'byte_identical_to_source':True}
        result['prefix']=prefix
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True,help='Reference-only calibration B6 run')
    p.add_argument('--conditioning',type=Path,required=True,help='Calibration conditioning cache')
    p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--contract',type=Path,default=ROOT/'tests/metal_reference_contract.json')
    p.add_argument('--acceptance',type=Path,default=ROOT/'tests/metal_fp16_performance_acceptance.json')
    p.add_argument('--quality-acceptance',type=Path,help='Explicit user acceptance; reuse its B6 and report perceptual screens without stopping')
    p.add_argument('--calibration-only',action='store_true',help='Stop after the first production-VAE paired B6')
    a=p.parse_args()
    if a.output.exists():p.error('use a fresh output directory')
    freeze=json.loads(a.freeze.read_text());check_hashes(freeze['sha256'])
    if not freeze['calibration_pass'] or freeze['candidate_B6_seen']:p.error('invalid pre-candidate calibration')
    if freeze['sha256'].get('tests/metal_reference_contract.json')!=sha(a.contract):p.error('contract is not frozen')
    contract=json.loads(a.contract.read_text());check_hashes(contract['source_assets_sha256'])
    acceptance=json.loads(a.acceptance.read_text());check_hashes(acceptance['evidence_sha256'])
    if acceptance['geometry']!=[640,480,243] or acceptance['maximum_evaluations']!=6:p.error('wrong acceptance scope')
    provenance=json.loads((a.binary.parent/'build-provenance.json').read_text())
    if provenance['binary_sha256']!=sha(a.binary):p.error('changed frozen binary')
    ref=validate_run(a.reference,False)
    accepted=ref['manifest']['binary_build_provenance']
    if not accepted or accepted['binary_sha256']!=acceptance['binary_sha256']:
        p.error('calibration reference does not bind the accepted executable')
    # The new executable adds diagnostic-only hooks. Do not carry the accepted
    # performance exception over to a changed attention implementation.
    for name in ('src/metal/fp16_attention.metal','src/metal/fp16_attention_host.inc','src/metal/routed_attention.metal',
                 'src/metal/native_attention.metal','src/metal/native_attention_host.inc'):
        if provenance['source_sha256'][name]!=accepted['source_sha256'][name]:
            p.error('accepted attention implementation changed: '+name)
    if sha(a.conditioning)!=json.loads((a.reference/'record.json').read_text())['conditioning_sha256_after']:
        p.error('changed calibration conditioning')
    approved=quality_acceptance(a.quality_acceptance,a.binary,a.contract) if a.quality_acceptance else None
    if approved and Path(approved[2]['reference']).resolve()!=a.reference.resolve():
        p.error('accepted B6 reference differs from this run')
    a.output.mkdir(parents=True)
    ledger={'version':1,'contract_sha256':sha(a.contract),'freeze_sha256':sha(a.freeze),
            'accepted_performance_sha256':sha(a.acceptance),'binary_sha256':sha(a.binary),
            'script_sha256':sha(__file__),'cases':[],'reference_qualified':False,
            'perceptual_observation':'automated screens only; no human observation claimed'}
    if approved:
        ledger.update(quality_acceptance_sha256=sha(a.quality_acceptance),quality_basis='user-accepted current recipe; frozen metrics retained',
                      accepted_prior_index=str(approved[1]),user_accepted_recipe=True)
        reused=dict(approved[2]);reused['quality']=os.path.relpath(approved[1].parent/reused['quality'],a.output)
        reused['quality_disposition']='accepted by user';reused['reused']=True
        ledger['cases'].append(reused)
    def save():
        (a.output/'index.json').write_text(json.dumps(ledger,indent=2)+'\n')
        body=['<!doctype html><meta charset="utf-8"><title>M2 243-frame review</title>',
              '<style>body{font:16px system-ui;background:#161920;color:#eee;max-width:1300px;margin:2rem auto}video{width:48%;margin-right:1%}a{color:#9cd3ff}pre{white-space:pre-wrap}</style>',
              '<h1>M2: 243-frame Reference review</h1><p>Left: original BF16/MPSGraph. Right: mixed FP16. Six evaluations, production VAE. No human judgment is asserted.</p>']
        if approved:body+=['<p>The user accepted the initial B6 quality. Frozen perceptual metrics remain diagnostic for this unchanged recipe; execution and protection checks remain mandatory.</p>']
        if ledger.get('stop_reason'):body+=['<p><strong>'+html.escape(ledger['stop_reason'])+'</strong></p>']
        for row in ledger['cases']:
            body+=['<h2>'+html.escape(row['id'])+'</h2>']
            for path in (row['reference'],row['candidate']):
                body+=['<video controls preload="metadata" src="'+html.escape(os.path.relpath(Path(path)/'result.mp4',a.output))+'"></video>']
            if row.get('quality'):body+=['<p><a href="'+html.escape(row['quality'])+'/review.html">Paired metrics and playback</a></p>']
            if row.get('quality'):body+=['<p>Automated screen: <strong>'+('PASS' if row['pass'] else 'FAIL')+'</strong>.</p>']
            body+=['<details><summary>Validation record</summary><pre>'+html.escape(json.dumps(row,indent=2))+'</pre></details>']
        body+=['<p><a href="index.json">Records and hashes</a></p>']
        (a.output/'review.html').write_text('\n'.join(body)+'\n')
    def run(command,log):
        with log.open('w') as stream:r=subprocess.run(list(map(str,command)),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        return r.returncode
    entries=[{'id':'calibration','seed':contract['calibration']['seed'],'prompt':contract['calibration']['prompt'],'face':False}]
    if approved:entries=[]
    if not a.calibration_only:entries+=contract['heldout']
    save()
    for entry in entries:
        name=entry['id'];directory=a.output/name;directory.mkdir()
        conditioning=a.conditioning if name=='calibration' else directory/'conditioning.h3cond'
        reference=a.reference if name=='calibration' else directory/'reference'
        candidate=directory/'fp16';source=a.reference/'result.h3av' if entry.get('context') else None
        command=[sys.executable,ROOT/'scripts/metal_native_bench.py','--binary',a.binary,
            '--case','B6','--full-vae','--conditioning',conditioning,'--seed',entry['seed'],'--prompt',entry['prompt']]
        for path in entry.get('images',[]):command+=['--reference',ROOT/path]
        for key,flag in (('video','ref-video'),('audio','ref-audio'),('first','first-frame'),('last','last-frame')):
            if key in entry:command+=['--'+flag,ROOT/entry[key]]
        if source:command+=['--continue-from',source,'--continue-context',entry['context']]
        row={'id':name,'reference':str(reference.resolve()),'candidate':str(candidate.resolve()),'pass':False}
        ledger['cases'].append(row);save()
        if name!='calibration' and run(command+['--output',reference,'--save-conditioning'],directory/'reference-driver.log'):
            row['error']='reference execution failed';save();raise SystemExit(2)
        flags=['--backend','metal','--metal-attention-kernel','steel-routed','--metal-attention-dtype','fp16','--metal-tier','reference']
        # Main B6 retains free-trajectory captures. Held-out cases additionally
        # diagnose conditioning-dependent ranges; their timings are not gates.
        diagnostics=['--capture-steps'] if name=='calibration' else ['--capture-ranges']
        if run(command+['--output',candidate]+flags+diagnostics,directory/'fp16-driver.log'):
            row['error']='candidate execution failed';save();raise SystemExit(2)
        row['hard_checks']=check_pair(reference,candidate,source)
        quality=directory/'quality';qc=[sys.executable,ROOT/'scripts/metal_reference_quality.py',reference/'result.mp4',
            '--candidate',candidate/'result.mp4','--contract',a.contract,'--output',quality]
        if entry.get('face'):qc+=['--face']
        if entry.get('context'):qc+=['--prefix',entry['context']]
        result=run(qc,directory/'quality-driver.log')
        row['quality']=str(quality.relative_to(a.output))
        row['record_sha256']={label:sha(path/'record.json') for label,path in (('reference',reference),('candidate',candidate))}
        if (quality/'metrics.json').exists():
            q=json.loads((quality/'metrics.json').read_text());row['metrics_sha256']=sha(quality/'metrics.json')
            row['pass']=result==0 and q.get('screen_pass',False)
        if result not in (0,3) or not (quality/'metrics.json').exists():
            row['error']='quality measurement failed';save();raise SystemExit(2)
        if approved:
            row['quality_disposition']='diagnostic under user acceptance; no separate human observation claimed'
        if not row['pass']:
            if not approved:
                ledger['stop_reason']='Frozen quality screen failed; remaining corpus is unqualified.';save();raise SystemExit(3)
        save()
    ledger['reference_qualified']=not a.calibration_only and all(row['pass'] for row in ledger['cases'])
    ledger['integration_validated']=not a.calibration_only
    if approved:ledger['user_accepted_validation_complete']=not a.calibration_only
    if a.calibration_only:
        ledger['stop_reason']=approved[0].get('remaining_corpus','Calibration pair complete; held-out corpus has not run.') if approved else 'Calibration pair complete; held-out corpus has not run.'
    save()


if __name__=='__main__':main()
