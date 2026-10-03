#!/usr/bin/env python3
"""Opt-in full-model M4 tests. Downloads are explicit README steps, never automatic."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fold_lora import fold, MANIFEST, sha256, TURBO_SHA, FoldError
from workflow import assemble, record_provenance

ROOT = Path(__file__).resolve().parents[2]
REALISM_SHA = 'acc529601d2da117fb81179e76c56e488a3beab1171659d305f04fa3655b787e'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'models/MiniMax-H3')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/lora-validation')
    parser.add_argument('--turbo',type=Path,default=ROOT/'lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors')
    parser.add_argument('--realism',type=Path,default=ROOT/'lora/downloads/h3-realism-people-t2v-i2v-r2v.safetensors')
    args=parser.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
    env['H3_CPU_SAMPLER']='1'
    results={}
    def run(name,cmd,success=True, phrase=None, environment=None):
        begin=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            p=subprocess.run([str(x) for x in cmd],cwd=ROOT,env={**env,**(environment or {})},stdout=log,stderr=log)
        text=(out/(name+'.log')).read_text()
        if (p.returncode==0)!=success or (phrase and phrase not in text):
            raise AssertionError(f'{name} failed ({p.returncode}): {text[-3000:]}')
        results[name]=dict(seconds=time.monotonic()-begin,returncode=p.returncode)
        print('PASS',name,flush=True)
        (out/'integration-results.json').write_text(json.dumps(results,indent=2)+'\n')
        return text
    for name,adapter,expected,modes in [('turbo',args.turbo,TURBO_SHA,['FL2VA']),
                                       ('realism',args.realism,REALISM_SHA,['FL2VA','Ref2VA'])]:
        assert sha256(adapter)==expected, f'unexpected published adapter identity: {adapter}'
        for mode in modes:
            directory=out/name/mode/'transformer'
            if not directory.exists():
                fold(args.base/mode/'transformer',[(adapter.resolve(),1.0)],directory)
            manifest=json.loads((directory/MANIFEST).read_text())
            assert manifest['adapters'][0]['sha256']==expected
            assert manifest['adapters'][0]['user_scale']==1.0
            assert len(manifest['adapters'][0]['targets'])==(259 if name=='turbo' else 104)
            assert all(t['dtype']=='BF16' for t in manifest['adapters'][0]['tensors'])
            assemble(args.base,out/name,mode)
            # Full source identity and output hashes, including untouched bytes.
            for item in manifest['base_files']:
                assert sha256(args.base/mode/'transformer'/item['file'])==item['sha256']
                assert sha256(directory/item['file'])==item['folded_sha256']
            results[f'fold-{name}-{mode}']=dict(pairs=len(manifest['adapters'][0]['targets']),
                modified=len(manifest['modified_tensors']),base_sha256=manifest['base_sha256'],
                source_unchanged=True,shards_verified=True)
    helper=ROOT/'bin/lora_fingerprint'
    helper.parent.mkdir(parents=True,exist_ok=True)
    run('build-fingerprint',['clang','-I.', 'lora/tests/fingerprint.c','bin/libh3.a',
                            '-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',
                            '-framework','MetalPerformanceShadersGraph','-framework','Accelerate','-licucore','-lm','-o',helper])
    identities={}
    for name,model,mode in [('base',args.base,'FL2VA'),('turbo',out/'turbo','FL2VA'),
                          ('realism',out/'realism','FL2VA'),('base-ref',args.base,'Ref2VA'),
                          ('realism-ref',out/'realism','Ref2VA')]:
        text=run('fingerprint-'+name,[helper,model,mode])
        identities[name]=text.strip().split()
    assert len({identities[n][0] for n in ('base','turbo','realism')})==3
    assert len({identities[n][1] for n in ('base','turbo','realism')})==1
    assert identities['base-ref'][0]!=identities['realism-ref'][0]
    assert identities['base-ref'][1]==identities['realism-ref'][1]
    results['fingerprints']=identities
    prompt='r34l1sm, the woman smiles and turns toward the camera. Steady camera, quiet room ambience.'
    def generate(name,model,steps=20,extra=(),state=True):
        cmd=[ROOT/'bin/h3cli','-d',model,'-p',prompt,'--seed','72','--width','128','--height','128',
             '--frames','56','--steps',str(steps),'-o',out/(name+'.mp4'),*extra]
        if state: cmd+=['--save-av-state',out/(name+'.h3av')]
        run(name,cmd)
        if state:
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',out/(name+'.mp4')]))
            assert {s['codec_type'] for s in probe['streams']}=={'audio','video'}
            video=next(s for s in probe['streams'] if s['codec_type']=='video')
            assert (video['width'],video['height'])==(128,128)
            results[name]['mp4_sha256']=sha256(out/(name+'.mp4'))
            results[name]['av_state_sha256']=sha256(out/(name+'.h3av'))
    face=['--first-frame',ROOT/'inputs/face1.jpg']
    generate('base-style',args.base,extra=face)
    generate('realism-style',out/'realism',extra=face)
    generate('realism-repeat',out/'realism',extra=face)
    assert sha256(out/'realism-style.mp4')==sha256(out/'realism-repeat.mp4')
    assert sha256(out/'base-style.mp4')!=sha256(out/'realism-style.mp4')
    generate('turbo-eight',out/'turbo',steps=8,extra=face)
    generate('turbo-repeat',out/'turbo',steps=8,extra=face)
    assert sha256(out/'turbo-eight.mp4')==sha256(out/'turbo-repeat.mp4')
    generate('realism-ref',out/'realism',steps=8,extra=['--ref-image',ROOT/'inputs/face1.jpg'])
    for mode in ('hard','bridge'):
        generate('continue-'+mode,out/'realism',steps=8,
                 extra=['--continue-from',out/'realism-style.h3av','--continue-mode',mode])
    # Continuation may intentionally switch LoRA variants without altering VAE compatibility.
    generate('continue-cross-model',out/'turbo',steps=8,extra=['--continue-from',out/'realism-style.h3av'])
    provenance=record_provenance(out/'realism-style.h3av',out/'realism','FL2VA')
    assert json.loads(provenance.read_text())['state_sha256']==sha256(out/'realism-style.h3av')
    generate('pause-preview',out/'turbo',steps=8,state=False,
             extra=[*face,'--stop-after-step','3','--save-sampler-state',out/'turbo.h3sample','--preview-on-stop'])
    run('resume',[ROOT/'bin/h3cli','-d',out/'turbo','--resume-sampler-state',out/'turbo.h3sample',
                  '-o',out/'resumed.mp4','--save-av-state',out/'resumed.h3av'])
    assert sha256(out/'resumed.h3av')==sha256(out/'turbo-eight.h3av')
    assert sha256(out/'resumed.mp4')==sha256(out/'turbo-eight.mp4')
    for name,model in [('base',args.base),('other-fold',out/'realism')]:
        run('reject-resume-'+name,[ROOT/'bin/h3cli','-d',model,'--resume-sampler-state',out/'turbo.h3sample',
                                   '-o',out/'must-not-exist.mp4'],False,'model content fingerprint mismatch')
    # Per-step denoised preview callback and the unchanged memory-budget machinery.
    generate('denoised-preview',out/'turbo',steps=6,state=False,extra=[*face,'--show'])
    run('memory-guards',[ROOT/'bin/memory_tests'])
    run('folded-memory-guard',[ROOT/'bin/h3cli','-d',out/'turbo','-p',prompt,
        '--width','128','--height','128','--frames','56','--steps','8',
        '-o',out/'memory-must-not-exist.mp4'],False,'safety floor',
        {'H3_TEST_MIN_AVAILABLE_MEMORY_BYTES':str(1<<60)})
    assert not (out/'memory-must-not-exist.mp4').exists()
    run('qwen-gqa',[ROOT/'bin/qwen_scaling_tests'])
    results['acceptance']=dict(deterministic_style=True,deterministic_turbo=True,
        different_from_base=True,resume_exact=True,continuation_cross_variant=True,
        caveat='128px smoke tests establish compatibility and reproducibility, not production-resolution visual quality.')
    (out/'integration-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('PASS all full-checkpoint LoRA integration tests',flush=True)


if __name__=='__main__': main()
