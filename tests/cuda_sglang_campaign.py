#!/usr/bin/env python3
"""Three serial, interleaved fresh-process pairs for every frozen primary case.

Failures remain in the campaign. Missing measurements are failed gates, never
zero durations. Operator tracing is disabled. CPU media checks run after each
pair, and their time is outside both render measurements.
"""
import argparse, hashlib, json, statistics, subprocess, sys, time
from pathlib import Path


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def measure(result,engine):
    v=result.get('validation',{});s=v.get('stage_seconds',{})
    return dict(wall=result.get('wall_seconds'),playable=result.get('complete_playable_seconds'),
        generation=v.get('generation_seconds'),
        denoising=s.get('MiniMaxH3DenoisingStage' if engine=='sglang' else 'denoise'),
        full_vae=s.get('MiniMaxH3DecodingStage') if engine=='sglang' else
            sum(s[k] for k in ('audio VAE','video VAE decode')) if all(k in s for k in ('audio VAE','video VAE decode')) else None,
        peak_vram=result.get('sampled_peak_gpu_bytes'),sample_gap=result.get('largest_sample_gap_seconds'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','out','oracle-python','metrics-site'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('reference-cublas','reference-cudnn','reference-jpeg','reference-ffmpeg'):p.add_argument('--'+name,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    contract_path=Path(__file__).with_name('cuda_sglang_contract.json');contract=json.loads(contract_path.read_text());gate=contract['gates']['performance'];pairs=[]
    def report():
        summaries={}
        for case in contract['primary']:
            case_pairs=[r for r in pairs if r['case']==case['id']];summary=dict(pairs=len(case_pairs),measurements={})
            for metric in ('wall','playable','generation','denoising','full_vae','peak_vram','sample_gap'):
                sides={engine:[r[engine]['measurements'].get(metric) for r in case_pairs if engine in r] for engine in ('sglang','native')}
                valid=all(len(x)==gate['interleaved_pairs_per_primary'] and all(isinstance(v,(int,float)) and v>=0 for v in x) for x in sides.values())
                m=dict(complete=valid)
                if valid:
                    m.update({k:dict(median=statistics.median(v),minimum=min(v),maximum=max(v),samples=v) for k,v in sides.items()})
                    m['native_over_oracle']=m['native']['median']/max(m['sglang']['median'],1e-12)
                    if metric=='sample_gap':m['passed']=max(sides['native']+sides['sglang'])<=gate['sampling_gap_seconds_max']
                    elif metric in ('generation','denoising','full_vae','peak_vram'):m['passed']=m['native_over_oracle']<=gate[metric+'_ratio_max']
                if metric not in ('wall','playable'):m.setdefault('passed',False)
                summary['measurements'][metric]=m
            summary['content_passed']=len(case_pairs)==3 and all(r.get('content_passed',False) for r in case_pairs)
            summary['passed']=summary['content_passed'] and all(m.get('passed',True) for m in summary['measurements'].values())
            summaries[case['id']]=summary
        value=dict(kind='matched fresh-process CUDA reference campaign',contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),pairs=pairs,summary=summaries,passed=all(s['passed'] for s in summaries.values()))
        write(a.out/'result.json',value);return value
    for repeat in range(gate['interleaved_pairs_per_primary']):
        for spec in contract['primary']:
            case=spec['id'];pair=dict(case=case,repeat=repeat+1);pairs.append(pair)
            order=('sglang','native') if repeat%2==0 else ('native','sglang')
            pair['order']=list(order)
            for engine in order:
                root=a.out/f'{engine}-{case}-r{repeat+1}'
                cmd=[sys.executable,str(a.source/'tests/cuda_sglang.py'),'--engine',engine,'--case',case,'--source',str(a.source),'--out',str(root)]
                if engine=='native':
                    cmd+=['--reference']
                    for name in ('reference_cublas','reference_cudnn','reference_jpeg','reference_ffmpeg'):cmd+=['--'+name.replace('_','-'),getattr(a,name)]
                write(a.out/(root.name+'-command.json'),dict(argv=cmd))
                with (a.out/(root.name+'.log')).open('x') as log:r=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                result=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else {}
                pair[engine]=dict(root=str(root),exit_code=r.returncode,status=result.get('status','missing'),measurements=measure(result,engine))
                report();print(root.name,pair[engine]['status'],flush=True)
            if all(pair[e]['status']=='media_valid' for e in ('native','sglang')):
                metrics=a.out/f'media-{case}-r{repeat+1}'
                cmd=[str(a.oracle_python),str(a.source/'tests/cuda_sglang_media.py'),str(Path(pair['sglang']['root'])/'video.mp4'),str(Path(pair['native']['root'])/'video.mp4'),str(metrics)]
                import os
                env=os.environ|dict(PYTHONPATH=str(a.metrics_site),OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
                with (a.out/(metrics.name+'.log')).open('x') as log:r=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
                pair['content_metrics']=str(metrics);pair['content_passed']=r.returncode==0
            report()
    return 0 if report()['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
