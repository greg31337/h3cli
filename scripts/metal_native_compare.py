#!/usr/bin/env python3
"""Compare retained B1/B5/B6 records and complete AV tensors; no new rendering."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('reference',type=Path)
p.add_argument('candidate',type=Path)
p.add_argument('--exact',action='store_true',help='Require byte-identical AV states for conditioning-cache validation')
p.add_argument('--allow-weight-format-change',action='store_true',help='Explicit BF16 vs Q8 ablation; other workload checks still apply')
p.add_argument('--max-relative-l2',type=float,default=.01)
p.add_argument('--min-cosine',type=float,default=.9999)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()

def load(directory):
    r=json.loads((directory/'record.json').read_text())
    if not r['evaluation_contract_pass']:raise SystemExit('Incomplete/invalid evaluations: '+str(directory))
    b=(directory/'result.h3av').read_bytes()
    if hashlib.sha256(b).hexdigest()!=r['av_sha256']:raise SystemExit('AV checksum changed: '+str(directory))
    if b[:8]!=b'H3AV\r\n\x1a\n':raise SystemExit('Unknown AV format')
    nv,na=struct.unpack_from('<QQ',b,72)
    if len(b)!=160+nv+na:raise SystemExit('AV payload size mismatch')
    return r,b,(nv,na)

r,rb,sizes=load(a.reference);c,cb,csizes=load(a.candidate)
if a.allow_weight_format_change:
    if r['backend']!='metal' or c['backend']!='metal' or r['attention']!=c['attention']:
        raise SystemExit('Q8 ablation requires the same Metal attention backend')
    left=dict(r['metal_options']);right=dict(c['metal_options'])
    left.pop('metal-weight-format');right.pop('metal-weight-format')
    if left!=right:raise SystemExit('Q8 ablation changed attention/kernel options')
for key in ('case','evaluations','blocks','geometry','weight_precision','sampler','seed','prompt','reference_sha256'):
    if key=='weight_precision' and a.allow_weight_format_change:
        if (r[key],c[key])!=('bf16','q8'):raise SystemExit('Expected explicit BF16 -> Q8 comparison')
        continue
    if r[key]!=c[key]:raise SystemExit('Workload mismatch: '+key)
for key in ('continuation_sha256','continuation_context','extra_inputs_sha256'):
    if r.get(key)!=c.get(key):raise SystemExit('Workload mismatch: '+key)
for key in ('limits_sha256','conditioning_identity','conditioning_sha256_after'):
    if r.get(key)!=c.get(key):raise SystemExit('Stale or mismatched benchmark input: '+key)
if sizes!=csizes or rb[24:128]!=cb[24:128]:raise SystemExit('AV identity differs')
def measure(x,y,name):
    xx=yy=xy=dd=0.;maximum=0.
    for f,g in zip(x,y):
        if not math.isfinite(f) or not math.isfinite(g):raise SystemExit('Nonfinite '+name+' tensor')
        delta=f-g;dd+=delta*delta;xx+=f*f;yy+=g*g;xy+=f*g;maximum=max(maximum,abs(delta))
    cosine=xy/math.sqrt(xx*yy) if xx*yy else float(xx==yy)
    rel=math.sqrt(dd/max(xx,1e-30))
    return {'relative_l2':rel,'cosine':cosine,'rmse':math.sqrt(dd/max(len(x),1)),'max_abs':maximum,
            'pass':rel<=a.max_relative_l2 and cosine>=a.min_cosine}

def floats(blob):
    values=array('f');values.frombytes(blob)
    if sys.byteorder!='little':values.byteswap()
    return values

metrics={};offset=160
geometry=struct.unpack_from('<10I',rb,24)
context=r.get('continuation_context')
for name,length in zip(('video','audio'),sizes):
    xbytes=rb[offset:offset+length];ybytes=cb[offset:offset+length];offset+=length
    metrics[name]=measure(floats(xbytes),floats(ybytes),name)
    if context:
        # A long unchanged prefix must not dilute generated-suffix error.
        k=(context-39)//51
        frames,spatial,prefix=(geometry[3],geometry[4]*geometry[5],12+15*k) if name=='video' else (geometry[6],1,65+85*k)
        stride=frames*spatial*4;cut=prefix*spatial*4
        if not 0<cut<stride or length%stride:raise SystemExit('Invalid continuation geometry')
        xp=b''.join(xbytes[c:c+cut] for c in range(0,length,stride))
        yp=b''.join(ybytes[c:c+cut] for c in range(0,length,stride))
        xs=b''.join(xbytes[c+cut:c+stride] for c in range(0,length,stride))
        ys=b''.join(ybytes[c+cut:c+stride] for c in range(0,length,stride))
        metrics[name+'_protected_prefix']={'byte_identical':xp==yp,'pass':xp==yp}
        metrics[name+'_suffix']=measure(floats(xs),floats(ys),name+' suffix')
ref_t=r['timing'];cand_t=c['timing']
result={'reference':str(a.reference),'candidate':str(a.candidate),'case':r['case'],
        'thresholds':{'max_relative_l2':a.max_relative_l2,'min_cosine':a.min_cosine},
        'byte_identical':rb==cb,'metrics':metrics,
        'denoise_speedup':ref_t['summed_denoise_seconds']/cand_t['summed_denoise_seconds'],
        'steady_speedup':ref_t['steady_median']/cand_t['steady_median'] if ref_t['steady_median'] and cand_t['steady_median'] else None,
        'timing_comparable':not(any(x.get(k) for x in (r,c) for k in ('component_fences','region_fences','capture_qkv','capture_boundaries','capture_steps','capture_ranges','teacher_from','trace_command')))}
result['pass']=all(m['pass'] for m in metrics.values()) and (not a.exact or rb==cb)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if not result['pass']:raise SystemExit('AV numerical gate failed')
