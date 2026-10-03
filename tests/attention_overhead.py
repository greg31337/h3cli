#!/usr/bin/env python3
"""Matched unprofiled target repeats, preserving arithmetic and latent equality."""
import argparse,json,statistics
from pathlib import Path
from attention_run import run,sha
from attention_workflows import compare_av

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--production',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
p.add_argument('--modes',nargs='+',default=['default','sage2++','sage3'])
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
ledger=json.loads((a.production/'ledger.json').read_text());results=[]
# The bounded workflow cache may have released this family's generated packs
# since the production matrix. Rewarm outside the measured unprofiled pairs.
reference=next(r for r in ledger if r['name']=='production-fl-turbo-nvfp4-42-r0-default')
if any(not (a.output/f'overhead-fl-turbo-nvfp4-{mode}-r{repeat}.json').exists() for mode in a.modes for repeat in range(2)):
    dest=a.output/'warmup';dest.mkdir(exist_ok=True)
    name='warmup-'+str(len(list(dest.glob('warmup-*.json'))))
    cmd=reference['argv'].copy();cmd.remove('--profile')
    for flag,value in [('--width','128'),('--height','128'),('--frames','22'),('--steps','2'),
                       ('--save-av-state',str(dest/(name+'.h3av'))),('-o',str(dest/(name+'.mp4')))]:
        cmd[cmd.index(flag)+1]=value
    env=reference['environment'].copy();env.pop('H3_PROFILE',None)
    warm=run(name,cmd,dest,3000,env)
    if warm['returncode']:raise RuntimeError('overhead cache warmup failed')
for mode in a.modes:
    profiled=[r for r in ledger if r['name'].startswith('production-fl-turbo-nvfp4-42-') and r['name'].endswith('-'+mode)]
    if len(profiled)!=2:raise RuntimeError('requires both matched profiled target repeats')
    samples=[]
    for repeat,reference in enumerate(profiled):
        name=f'overhead-fl-turbo-nvfp4-{mode}-r{repeat}';command=reference['argv'].copy();command.remove('--profile')
        command[command.index('-o')+1]=str(a.output/(name+'.mp4'))
        av=a.output/(name+'.h3av');command[command.index('--save-av-state')+1]=str(av)
        record=a.output/(name+'.json')
        if record.exists():
            row=json.loads(record.read_text())
            if row['argv']!=command or row['binary_sha256']!=sha(command[0]) or row['returncode']:raise RuntimeError('changed or failed overhead record')
        else:
            environment=reference['environment'].copy();environment.pop('H3_PROFILE',None)
            row=run(name,command,a.output,3000,environment)
        if row['returncode']:raise RuntimeError('unprofiled render failed')
        source=reference['argv'][reference['argv'].index('--save-av-state')+1]
        equality=compare_av(Path(source),av)
        if not all(v['exact'] for v in equality.values()):raise RuntimeError('profiling changed AV latents')
        samples.append({'profiled':reference,'unprofiled':row,'latent_equality':equality})
    result={'mode':mode,'samples':samples,
        'profiled_wall_mean':statistics.mean(x['wall_seconds'] for x in profiled),
        'unprofiled_wall_mean':statistics.mean(x['unprofiled']['wall_seconds'] for x in samples),
        'timing_scope':'Per-process wall and externally observed denoise-to-audio-VAE interval; inspect recorded polling uncertainty.'}
    if all('observed_denoising_seconds' in x['profiled'] and 'observed_denoising_seconds' in x['unprofiled'] for x in samples):
        result['profiled_observed_denoise_mean']=statistics.mean(x['profiled']['observed_denoising_seconds'] for x in samples)
        result['unprofiled_observed_denoise_mean']=statistics.mean(x['unprofiled']['observed_denoising_seconds'] for x in samples)
    results.append(result);(a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
