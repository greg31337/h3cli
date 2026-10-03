#!/usr/bin/env python3
"""Finite calibration: three frozen floors, identical teacher-forced inputs."""
import fcntl,json,re,sys,time
from pathlib import Path
import numpy as np
from cuda_sol_campaign import ROOT,ACCEPT,render,execute,save,verify,spent
from cuda_sol_oracle import metrics

def main():
    with (ROOT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        spec=json.loads(ACCEPT.read_text())['calibration']
        base=dict(id='C-calibration',width=640,height=480,frames=243,steps=2,seed=spec['seed'],prompt=spec['prompt'],references=[],image_size='match',diagnostics=True,capture=True)
        dense=render(base,'default',.75,'C',750,dict(H3_CPU_SAMPLER='1'))
        if dense['status']!='pass':raise RuntimeError('Dense calibration failed')
        teacher=ROOT/'diagnostics/C-calibration-default/steps';rows=[]
        for minimum in spec['minimums']:
            c=dict(base,id='C-min-'+str(minimum),capture=False)
            row=render(c,'sol',minimum,'C',min(650,2100-spent('C')),dict(H3_CPU_SAMPLER='1',H3_TEST_NATIVE_TEACHER_DIR=str(teacher)))
            candidate=ROOT/'diagnostics'/(c['id']+'-sol')/'steps';quality={}
            if row['status']=='pass':
                for modality in ['video','audio']:
                    x=np.fromfile(candidate/f'step-002-{modality}-velocity.f32',dtype='<f4');y=np.fromfile(teacher/f'step-002-{modality}-velocity.f32',dtype='<f4');quality[modality]=metrics(x,y)
            passed=row['status']=='pass' and all(m['finite'] and m['relative_l2']<=.05 and m['cosine']>=.998 for m in quality.values()) and len(quality)==2
            rows.append(dict(minimum=minimum,screen_pass=passed,quality=quality,wall_seconds=row['wall_seconds'],step_seconds=row['step_seconds']))
            save(ROOT/'calibration.json',rows)
        winners=[r for r in rows if r['screen_pass']]
        selected=min(winners,key=lambda r:r['step_seconds'][-1] if r['step_seconds'] else r['wall_seconds']) if winners else dict(minimum=.75,screen_pass=False,reason='No frozen preset passed conservative teacher-forced velocity gates')
        selected['selection']='Fastest screen-passing finite preset; frozen before held-out comparisons';save(ROOT/'candidate.json',selected)
        # Operator correctness/timings on both independently captured dense QKV tensors.
        files=sorted((ROOT/'diagnostics/C-calibration-default/qkv').glob('*.qkv'))
        for i,path in enumerate(files):
            seq=int(re.search(r'-s-(\d+)',path.name)[1]);out=ROOT/'real-replay'/f'block-{i}'
            execute('C-oracle-'+str(i),'C',[sys.executable,'tests/cuda_sol_oracle.py','--binary','./bin/cuda_sol_native','--output',str(out),'--input',str(path),'--layout',str(path.parent/'layout.bin'),'--sequence',str(seq),'--minimum',str(selected['minimum'])],min(180,2100-spent('C')),allow_failure=True)
            execute('C-dense-replay-'+str(i),'C',['./bin/attention_native','default',str(seq),str(out/'dense.bf16'),'0','5',str(path)],min(120,2100-spent('C')),dict(H3_TEST_ATTENTION_FAST='0'),allow_failure=True)
            native=json.loads((out/'real/result.json').read_text())
            argv=native['argv'];argv[argv.index('--exact')+1]='1'
            for key,name in [('--output','forced-exact.bf16'),('--routes','forced-routes.bin'),('--stats','forced-native.json')]:argv[argv.index(key)+1]=str(out/name)
            execute('C-forced-replay-'+str(i),'C',argv,min(180,2100-spent('C')),allow_failure=True)
        print(json.dumps(selected),flush=True)
if __name__=='__main__':main()
