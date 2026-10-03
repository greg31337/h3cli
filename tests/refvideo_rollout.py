#!/usr/bin/env python3
"""Emit a rollout decision only when every retained numerical/render gate passes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation',type=Path,required=True)
    args=p.parse_args();out=args.validation
    paths={'encoder':out/'native/results.json','dit':out/'dit-small-native-precision/results.json',
        'adaln':out/'adaln-results.json',
        'generation':out/'generation-results.json','legacy':out/'legacy-renders/results.json',
        'ab':out/'ab/results.json','ab_review':out/'ab/review.json'}
    result={'pipeline':'released-v1','production_ready':False,'failed_gates':[],'evidence':{}}
    decision=out/'rollout.json'
    # Invalidate an earlier approval before checking any possibly newer evidence.
    decision.write_text(json.dumps(result,indent=2)+'\n')
    records={}
    for name,path in paths.items():
        if not path.exists():
            result['failed_gates'].append(name+': missing evidence')
            continue
        records[name]=json.loads(path.read_text())
        result['evidence'][name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    required={'raw-raw17','raw-tiled','stitch-stitch','temporal-temporal39-tiled','released-temporal39-tiled'}
    required|={f'{mode}-temporal{n}' for mode in ('temporal','released') for n in (39,48,56,60,73,80,107,110,124)}
    for name,d in records.items():
        if name=='encoder':ok=d['passed'] and d['legacy_exact'] and required<={c['case'] for c in d['cases']}
        elif name=='dit':ok=d['passed'] and d['oracle']['full'] and {'video_velocity','audio_velocity','video_final','audio_final'}<=d['cases'].keys()
        elif name=='generation':ok=d['passed'] and d['oracle_generation']['full']
        elif name=='adaln':ok=d['passed']
        elif name=='legacy':ok=all(d.get(k,{}).get('passed',False) for k in ('video','images','mixed'))
        elif name=='ab':ok=all(d.get(k,{}).get('video_audio_timing_passed',False) for k in ('video','mixed'))
        else:
            ab=records.get('ab',{})
            ok=bool(d.get('soundtrack') and d.get('limitations')) and all(
                d.get(k,{}).get('conclusion') and
                d.get('artifact_sha256',{}).get(k,{}).get('released')==ab.get(k,{}).get('sha256')
                for k in ('video','mixed'))
        if not ok:result['failed_gates'].append(name)
    # Do not merely trust a manually edited status document for the cheap gates.
    for binary in ('bin/video_posterior_tests','bin/refvideo_layout_tests'):
        if subprocess.run([str(ROOT/binary)],cwd=ROOT).returncode:
            result['failed_gates'].append(binary)
    result['production_ready']=not result['failed_gates']
    result['scope']='M4 conditioning and controlled Qwen/noise generation; not cross-RNG pixel identity'
    decision.write_text(json.dumps(result,indent=2)+'\n')
    if result['failed_gates']:raise SystemExit('HOLD released-v1: '+', '.join(result['failed_gates']))
    print('PASS released-v1 rollout gates',flush=True)
if __name__=='__main__':main()
