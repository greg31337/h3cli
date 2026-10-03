#!/usr/bin/env python3
"""One tiny resume retry after explicitly timed cold fingerprint preparation."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import fcntl,json
from pathlib import Path
from cuda_sol_campaign import MODEL,ROOT,execute,verify,save,media,command
from cuda_sol_checks import av

def main():
    with (ROOT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);identity=verify()
        execute('E-fingerprint-build','E',['cc','-std=c11','-D_GNU_SOURCE','-DH3_CUDA_USE_SOL','-I.',
            'tests/cuda_sol_fingerprint.c','bin/libh3.a','-L/usr/local/cuda/lib64',
            '-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-lcublas','-lcublasLt',
            '-lstdc++','-licuuc','-ljson-c','-lcrypto','-lpthread','-lm','-o','bin/cuda_sol_fingerprint'],60)
        execute('E-cold-fingerprint','E',['./bin/cuda_sol_fingerprint',str(MODEL)],600)
        prior=json.loads((ROOT/'runs/E-uninterrupted-sol/record.json').read_text())
        tiny=prior['case'];minimum=prior['minimum'];checkpoint=ROOT/'paused-retest.h3sample'
        paused=dict(tiny,id='E-paused-retest',save_state=False,
                    extra=['--stop-after-step','1','--save-sampler-state',str(checkpoint)])
        execute('E-paused-retest-sol','E',command(paused,'sol',ROOT/'runs/E-paused-retest-sol',minimum,identity),240,monitor=True)
        assert checkpoint.is_file()
        resumed=ROOT/'runs/E-resumed-retest-sol'
        execute('E-resumed-retest-sol','E',['./bin/h3cli','-d',str(MODEL),
            '--resume-sampler-state',str(checkpoint),'--save-av-state',str(resumed/'output.h3av'),
            '-o',str(resumed/'output.mp4')],240,monitor=True)
        x,y=av(ROOT/'runs/E-uninterrupted-sol/output.h3av');xx,yy=av(resumed/'output.h3av')
        equality=dict(video_exact=x.tobytes()==xx.tobytes(),audio_exact=y.tobytes()==yy.tobytes())
        save(ROOT/'resume-retest-equivalence.json',equality);assert all(equality.values())
        assert media(resumed,tiny)['valid']
        for label,flags in [('conflict',['--sol-min-exact','0.123']),('default',['--resume-default-cuda'])]:
            ident='E-resume-retest-reject-'+label
            row=execute(ident,'E',['./bin/h3cli','-d',str(MODEL),'--resume-sampler-state',str(checkpoint)]+flags,60,allow_failure=True)
            log=(ROOT/'runs'/ident/'stderr.log').read_text()
            result=dict(rejected=row['returncode']!=0 and ('resume attention differs' in log or 'generation-changing arguments' in log),checkpoint_exists=checkpoint.is_file(),message=log)
            save(ROOT/('resume-retest-rejection-'+label+'.json'),result);assert result['rejected']
        save(ROOT/'resume-retest.json',dict(paused='E-paused-retest-sol',resumed='E-resumed-retest-sol',
            replaces=['E-paused-sol','E-resumed-sol','E-resume-reject-conflict','E-resume-reject-default'],
            reason='Original pause hit its 240-second cap during cold full-model fingerprinting, before denoising. Preparation was timed separately; rendering caps, binary and criteria are unchanged.'))
if __name__=='__main__':main()
