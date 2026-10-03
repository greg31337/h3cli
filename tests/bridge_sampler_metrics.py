#!/usr/bin/env python3
"""Audit real bridge CPU/GPU trajectories, reuse, current heads and AV output.

Require bitwise equality, which is stricter than the repository's existing
DiT numerical bounds. Approximate reuse is compared with the equivalent CPU
reuse configuration, not mistaken for an exact substitute for reuse=1.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from bridge_metrics import state, diagnostics


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def sidecar(base,ext): return Path(str(base)+'.'+ext)


def difference(a,b):
    a=np.asarray(a); b=np.asarray(b); delta=a.astype(np.float64)-b
    return dict(bit_identical=a.tobytes()==b.tobytes(),maximum_absolute=float(np.max(np.abs(delta))),
                relative_l2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(a.ravel()),1e-20)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args(); out=args.directory.resolve()
    manifests=list(out.glob('*-run.json')); assert len(manifests)==1
    run=json.loads(manifests[0].read_text()); assert run['passed']
    assert sha(sidecar(Path(run['source']),'h3av'))==run['source_sha256']
    assert set(run['outputs'])==set(run['variants'])
    log=manifests[0].with_name(manifests[0].name.replace('-run.json','.log')).read_text()
    cases={}; cursor=0
    for name,config in run['variants'].items():
        base=out/name
        for ext,value in run['outputs'][name].items(): assert sha(sidecar(base,ext))==value,(name,ext)
        end=log.index('\n',log.index(f'ok: {base},',cursor)); excerpt=log[cursor:end]; cursor=end
        saved=state(sidecar(base,'h3av')); meta=json.loads(sidecar(base,'json').read_text())
        assert meta['steps']==20 and meta['frames']==51 and meta['audio_samples']==68000
        assert sidecar(base,'rgb').stat().st_size==51*meta['width']*meta['height']*3
        pcm=np.fromfile(sidecar(base,'pcm'),'<f4')
        assert pcm.size==2*68000 and np.isfinite(pcm).all()
        assert meta['reuse']==config['reuse'] and meta['core_reuse']==config['core']
        assert meta['gpu_requested']==config['gpu']
        assert f'bridge continuation uses {"GPU-state F32" if config["gpu"] else "CPU F32"} Euler sampler' in excerpt
        trace=None
        if config['callbacks']:
            assert meta['latent_callbacks']==21
            trace=np.fromfile(sidecar(base,'trace'),'<f4').reshape(21,-1)
            assert trace.shape[1]==saved['video'].size+saved['audio'].size and np.isfinite(trace).all()
            final=np.concatenate([saved['video'].ravel(),saved['audio'].ravel()])
            assert trace[-1].tobytes()==final.tobytes()
            masks=np.asarray([float(v) for v in re.search(r'video bridge masks:([^\n]+)',excerpt)[1].split()])
            exact=np.flatnonzero(masks==0)
            video=trace[:,:saved['video'].size].reshape(21,*saved['video'].shape)
            for step in range(1,21): assert video[step,:,exact].tobytes()==video[0,:,exact].tobytes()
            begin,end_audio=map(int,re.search(r'audio bridge ticks=\[0,(\d+)\), exact=\[\d+,(\d+)\)',excerpt).groups())
            audio=trace[:,saved['video'].size:].reshape(21,*saved['audio'].shape)
            for step in range(1,21): assert audio[step,:,:,begin:end_audio].tobytes()==audio[0,:,:,begin:end_audio].tobytes()
        else: assert meta['latent_callbacks']==0
        if config['gpu']:
            audit=[int(s) for s in re.findall(r'GPU bridge exact video/audio bits unchanged through step (\d+)',excerpt)]
            assert audit and audit[-1]==20
            if config['callbacks']: assert audit==list(range(1,21))
        else: diagnostics(excerpt,20)
        core_count=20 if config['reuse']==1 else len({0,19,*range(0,20,config['reuse'])})
        if config['core']>1:
            heads=re.findall(r'bridge core audit step=(\d+) evaluated=(\d+) heads=current classes=([^\n]+)',excerpt)
            assert [int(s) for s,_,_ in heads]==list(range(20))
            evaluated=[int(s) for s,e,_ in heads if e=='1']
            assert evaluated==sorted({0,19,*range(0,20,config['core'])})
            assert all(len(classes.split())>=4 for _,_,classes in heads)
            core_count=len(evaluated)
        mux=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(sidecar(base,'mp4'))]))['streams']
        assert {s['codec_type'] for s in mux}=={'video','audio'}
        assert all(float(s['start_time'])==0 and abs(float(s['duration'])-51/24)<.001 for s in mux)
        assert next(s for s in mux if s['codec_type']=='video')['nb_frames']=='51'
        cases[name]=dict(saved=saved,trace=trace,config=config,seconds=meta['seconds'],core_evaluations=core_count)
    comparisons={}
    for name,case in cases.items():
        if not case['config']['gpu']: continue
        partners=[(n,c) for n,c in cases.items() if not c['config']['gpu'] and
                  all(c['config'][k]==case['config'][k] for k in ['reuse','core','length','strength','profile'])]
        assert len(partners)==1
        cpu_name,cpu=partners[0]
        comparison={stream:difference(cpu['saved'][stream],case['saved'][stream]) for stream in ['video','audio']}
        if case['trace'] is not None: comparison['trajectory']=difference(cpu['trace'],case['trace'])
        comparison['decoded_outputs_identical']={ext:run['outputs'][name][ext]==run['outputs'][cpu_name][ext]
                                                 for ext in ['rgb','pcm']}
        comparison.update(cpu=cpu_name,cpu_seconds=cpu['seconds'],gpu_seconds=case['seconds'],core_evaluations=case['core_evaluations'])
        comparisons[name]=comparison
    assert len(comparisons)==7, 'exact/reuse 2/3/core 4/6 and two callback-free GPU windows are required'
    passed=all(v['bit_identical'] for c in comparisons.values() for k,v in c.items() if k in ['video','audio','trajectory'])
    passed=passed and all(all(c['decoded_outputs_identical'].values()) for c in comparisons.values())
    report=dict(passed=passed,requirement='Bitwise CPU/GPU agreement at equivalent settings',
                source_sha256=run['source_sha256'],binary_sha256=run['binary_sha256'],shader_sha256=run['shader_sha256'],comparisons=comparisons)
    (out/'sampler-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert passed, 'CPU/GPU bridge regression failed; inspect sampler-results.json'


if __name__=='__main__': main()
