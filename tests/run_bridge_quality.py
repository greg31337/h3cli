#!/usr/bin/env python3
"""Paired real-model action tests and deterministic bridge parameter sweeps.

Every batch shares source, prompt, seed, reference conditioning and all 20
steps. Frozen executables/shaders keep a running sweep isolated from edits.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = 'The woman has the face in <Picture 1> and wears the outfit in <Picture 2>. '
SCENE = ('Wide full-body shot, her entire body from head to shoes stays visible. '
         'A locked-off camera watches the same sunlit garden path from a distance, '
         'with the same trees and lighting. Realistic continuous motion, quiet outdoor ambience. ')
SOURCE_PROMPT = ('Wide full-body shot, the entire woman from head to shoes stays visible, walking slowly along a straight garden path. '
                 + IDENTITY + 'Her arms swing naturally down beside her body. A locked-off camera sees her from a distance, '
                 'with a large stable garden and trees around her. She remains fully in frame. '
                 'Natural daylight, realistic continuous motion, quiet outdoor ambience.')
PROMPTS = {
    'unchanged': SOURCE_PROMPT,
    'look-up': IDENTITY + 'She continues her slow walking step and gradually looks upward while walking. ' + SCENE,
    'run': IDENTITY + 'She completes her current walking step, slows briefly, turns to her left across the path, '
           'and begins running to the left, staying entirely in frame. ' + SCENE,
    'arms': IDENTITY + 'She completes her walking step, gradually stops, and smoothly raises both arms from her sides above her head. ' + SCENE,
    'turn-left': IDENTITY + 'She slows to finish her current step, gradually turns to her left, then walks left across the path. ' + SCENE,
}
ACTION_NAMES=tuple(PROMPTS)
PROMPTS.update({
    'chain-run': PROMPTS['run'],
    'chain-walk': IDENTITY+'She completes her running stride, gradually slows to a walk, then turns toward the camera. '+SCENE,
    'chain-arms': IDENTITY+'She completes her walking step, gradually stops, and smoothly raises both arms from her sides. '+SCENE,
    'chain-pose-raise': IDENTITY+'She finishes her current walking step and stops moving forward. '
        'She stays at that position and gradually raises both arms from her sides above her head. '+SCENE,
    'chain-pose-lower': IDENTITY+'She remains standing at the same position with her feet planted. '
        'She slowly lowers both arms from overhead back to her sides, then gradually looks to her left. '+SCENE,
    'chain-pose-clasp': IDENTITY+'She remains standing at the same position with her feet planted. '
        'She finishes looking left, gradually turns her head back toward the camera, '
        'then smoothly brings both hands together in front of her chest. '+SCENE,
})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sidecar(base, extension):
    return Path(str(base)+'.'+extension)


def configuration(length=8, strength=.5, profile='stepped', mode='bridge', reuse=1, core=1, gpu=0, keep=0, window=1, callbacks=1):
    return dict(mode=mode, profile=profile, length=length, strength=strength, reuse=reuse, core=core, gpu=gpu, keep=keep,window=window,callbacks=callbacks)


def config_name(c):
    return 'hard' if c['mode'] == 'hard' else f"{c['profile']}-{c['length']}-{c['strength']:.2f}"


def sweep(full=False):
    if full:
        triples = itertools.product([4,6,8,9,10],[.25,.4,.5,.65],['stepped','linear','ease-out'])
    else:
        # Declared before observing any quality results: all main axes plus
        # selected interactions. --full expands this to the 60-cell grid.
        triples = [(8,.5,'stepped')] + [(n,.5,'stepped') for n in [4,6,9,10]]
        triples += [(8,m,'stepped') for m in [.25,.4,.65]]
        triples += [(8,.5,p) for p in ['linear','ease-out']]
        triples += [(6,.4,p) for p in ['stepped','linear','ease-out']]
        triples += [(9,.65,p) for p in ['linear','ease-out']]
        triples += [(4,.25,'linear'),(10,.25,'ease-out')]
    return [configuration(n,m,p) for n,m,p in triples]


def frozen_driver(out):
    binary = ROOT/'bin/continuation_generate'; shader = ROOT/'src/metal/shaders.metal'
    identity = sha(binary)+sha(shader)
    directory = out/('build-'+hashlib.sha256(identity.encode()).hexdigest()[:16])
    directory.mkdir(exist_ok=True)
    (directory/'src/metal').mkdir(parents=True,exist_ok=True)
    (directory/'bin').mkdir(exist_ok=True)
    for source, name in [(binary,'bin/driver'),(shader,'src/metal/shaders.metal')]:
        target=directory/name
        if not target.exists(): shutil.copy2(source,target)
        assert sha(source)==sha(target)
    for name in ['inputs','outputs']:
        target=directory/name
        if not target.exists(): target.symlink_to(ROOT/name,target_is_directory=True)
    return directory


def run_batch(out, build, source, action, seed, variants, model, resume=False, trace=False):
    tag=f'{action}-{seed}'
    manifest=out/(tag+'-run.json')
    prompt=PROMPTS[action] if action in PROMPTS else action
    names=[f'{tag}-{label}' for label,_ in variants]
    tsv=out/(tag+'.tsv')
    lines=['\t'.join(map(str,[out/name,c['mode'],c['profile'],c['length'],c['strength'],c['reuse'],c['core'],c['gpu'],c['keep'],c['window'],c['callbacks']]))
           for name,(_,c) in zip(names,variants)]
    expected=dict(prompt=prompt,seed=seed,source=str(source),source_sha256=sha(sidecar(source,'h3av')),
                  source_artifacts={ext:sha(sidecar(source,ext)) for ext in ['h3av','rgb','json']},
                  binary_sha256=sha(build/'bin/driver'),shader_sha256=sha(build/'src/metal/shaders.metal'),
                  references={name:sha(ROOT/'inputs'/name) for name in ['face1.jpg','body1.jpg']},
                  variants={name:c for name,(_,c) in zip(names,variants)})
    prior=json.loads(manifest.read_text()) if resume and manifest.exists() else {}
    if prior.get('passed') and set(prior.get('outputs',{}))==set(names) and all(prior.get(k)==v for k,v in expected.items()):
        if all((out/(name+'.'+ext)).exists() and sha(out/(name+'.'+ext))==value
               for name,hashes in prior['outputs'].items() for ext,value in hashes.items()):
            print('verified completed batch',tag,flush=True); return
    # Metadata is the completion marker. Remove stale markers before rerunning
    # so an old artifact cannot be mistaken for the new process's result.
    for name in names: sidecar(out/name,'json').unlink(missing_ok=True)
    tsv.write_text('\n'.join(lines)+'\n')
    command=[str(build/'bin/driver'),str(model),str(out/names[0]),'image1',str(sidecar(source,'h3av')),str(seed),
             '90','20','1','0',prompt,str(ROOT/'outputs/continuation-validation/reference.mp4'),'39','256']
    record=dict(expected,passed=False,command=command,outputs={},status='running')
    def save():
        temp=manifest.with_suffix('.tmp'); temp.write_text(json.dumps(record,indent=2)+'\n'); temp.replace(manifest)
    save()
    env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
    env.update(H3_TEST_VARIANTS=str(tsv),H3_PROFILE='1')
    if trace: env['H3_TEST_LATENT_TRACE']='1'
    started=time.monotonic(); print('running',tag,len(variants),'variants',flush=True)
    with (out/(tag+'.log')).open('w') as log:
        process=subprocess.Popen(command,cwd=build,env=env,stdout=log,stderr=log)
        while True:
            code=process.poll()
            for name in names:
                if name in record['outputs'] or not (out/(name+'.json')).exists(): continue
                try: meta=json.loads((out/(name+'.json')).read_text())
                except json.JSONDecodeError: continue
                callbacks=expected['variants'][name]['callbacks']
                assert meta['steps']==20 and meta['latent_callbacks']==(21 if callbacks else 0)
                record['outputs'][name]={ext:sha(out/(name+'.'+ext)) for ext in ['h3av','rgb','pcm','mp4','json']+(['trace'] if trace and callbacks else [])}
                print('completed',name,round(meta['seconds'],2),'seconds',flush=True); save()
            if code is not None: break
            time.sleep(1)
    record.update(returncode=code,seconds=time.monotonic()-started,status='complete' if code==0 else 'failed')
    record['conditioning_cache_hits']=(out/(tag+'.log')).read_text().count('conditioning cache hit')
    record['passed']=code==0 and len(record['outputs'])==len(names) and record['conditioning_cache_hits']==len(variants)-1
    assert all(sha(sidecar(source,ext))==value for ext,value in expected['source_artifacts'].items()), 'borrowed source modified'
    assert all(sha(ROOT/'inputs'/name)==value for name,value in expected['references'].items()), 'references modified'
    save()
    assert record['passed'], f'batch {tag} failed: inspect {out/(tag+".log")}'


def generate_source(out,build,source,model,seed):
    source.parent.mkdir(parents=True,exist_ok=True)
    command=[str(build/'bin/driver'),str(model),str(source),'image1','-',str(seed),'90','20','1','0',SOURCE_PROMPT]
    env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
    env.update(H3_CPU_SAMPLER='1',H3_PROFILE='1')
    with sidecar(source,'log').open('w') as log:
        subprocess.run(command,cwd=build,env=env,stdout=log,stderr=log,check=True)
    meta=json.loads(sidecar(source,'json').read_text())
    assert meta['frames']==90 and meta['steps']==20 and meta['latent_callbacks']==21
    record=dict(command=command,seed=seed,prompt=SOURCE_PROMPT,passed=True,
                binary_sha256=sha(build/'bin/driver'),shader_sha256=sha(build/'src/metal/shaders.metal'),
                references={n:sha(ROOT/'inputs'/n) for n in ['face1.jpg','body1.jpg']},
                outputs={ext:sha(sidecar(source,ext)) for ext in ['h3av','rgb','pcm','json','mp4']})
    sidecar(source,'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print('generated source',source,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/bridge-quality/acceptance')
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/bridge-quality/source',help='Base path with .h3av/.rgb/.json sidecars')
    parser.add_argument('--model',type=Path,default=ROOT/'models/MiniMax-H3')
    parser.add_argument('--phase',choices=['source','sweep','actions','samplers','chain'],default='sweep')
    parser.add_argument('--chain-style',choices=['pose','locomotion'],default='pose',
                        help='Stationary pose changes keep the character visible; locomotion tests leaving/returning to a locked camera')
    parser.add_argument('--full',action='store_true',help='All 60 sweep configurations instead of the declared 17-cell pilot')
    parser.add_argument('--only',help='Comma-separated action names')
    parser.add_argument('--seed',type=int,default=72)
    parser.add_argument('--length',type=int,default=8)
    parser.add_argument('--strength',type=float,default=.5)
    parser.add_argument('--profile',choices=['stepped','linear','ease-out'],default='stepped')
    parser.add_argument('--selection',type=Path,help='Use the fixed length/strength/profile recorded after a sweep')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--list',action='store_true')
    args=parser.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    if args.selection:
        selected=json.loads(args.selection.read_text())
        assert selected['protocol_sha256']==sha(ROOT/'docs/bridge-quality-protocol.json')
        args.length=selected['config']['length']; args.strength=selected['config']['strength']; args.profile=selected['config']['profile']
    hard=configuration(mode='hard'); chosen=configuration(args.length,args.strength,args.profile)
    if args.phase=='sweep': batches=[('run',[('hard',hard)]+[(config_name(c),c) for c in sweep(args.full)])]
    elif args.phase=='actions': batches=[(a,[('hard',hard),(config_name(chosen),chosen)]) for a in ACTION_NAMES]
    elif args.phase=='chain':
        actions=['chain-pose-raise','chain-pose-lower','chain-pose-clasp'] if args.chain_style=='pose' else ['chain-run','chain-walk','chain-arms']
        batches=[(a,[(config_name(chosen),chosen)]) for a in actions]
    elif args.phase=='samplers':
        variants=[('cpu',chosen),('gpu',dict(chosen,gpu=1))]
        variants += [(f'{"gpu" if gpu else "cpu"}-reuse-{n}',dict(chosen,gpu=gpu,reuse=n)) for n in [2,3] for gpu in [0,1]]
        variants += [(f'{"gpu" if gpu else "cpu"}-core-{n}',dict(chosen,gpu=gpu,core=n)) for n in [4,6] for gpu in [0,1]]
        variants += [(f'gpu-window-{window}-no-callback',dict(chosen,gpu=1,reuse=3,window=window,callbacks=0)) for window in [1,0]]
        batches=[('run',variants)]
    else: batches=[]
    if args.only:
        wanted=set(args.only.split(',')); batches=[b for b in batches if b[0] in wanted]
        assert wanted=={b[0] for b in batches}
    if args.phase=='chain': assert len(batches)>=2, 'chain validation requires at least three segments'
    if args.list:
        print(json.dumps(batches,indent=2)); return
    subprocess.run(['make','-j8','bin/continuation_generate'],cwd=ROOT,check=True)
    build=frozen_driver(out)
    source=args.source.resolve()
    if source.suffix=='.h3av': source=source.with_suffix('')
    if args.phase=='source':
        generate_source(out,build,source,args.model.resolve(),args.seed); return
    chain=[]
    for action,variants in batches:
        run_batch(out,build,source,action,args.seed,variants,args.model.resolve(),args.resume,args.phase=='samplers')
        if args.phase=='chain':
            target=out/f'{action}-{args.seed}-{variants[0][0]}'
            chain.append(dict(source=str(source),target=str(target),source_sha256=sha(sidecar(source,'h3av')),
                              target_sha256=sha(sidecar(target,'h3av')),prompt=PROMPTS[action]))
            source=target
    if chain:
        assert len(chain)>=2, 'chain validation requires at least three segments'
        (out/'chain.json').write_text(json.dumps(dict(passed=True,segments=len(chain)+1,style=args.chain_style,boundaries=chain),indent=2)+'\n')


if __name__=='__main__': main()
