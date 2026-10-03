#!/usr/bin/env python3
"""Replay selected/full final-head GEMMs on the same retained oracle inputs."""
import argparse,json,os,struct,subprocess
from pathlib import Path
from cuda_sglang_compare import oracle,metric
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--step',type=int,default=1)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--cublas',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);capture=a.oracle/'capture'
    prefix=f'step-{a.step:03d}.final_layer.audio_out.'
    x=oracle(capture,prefix+'input.0');y=oracle(capture,prefix+'output.0')
    indices=oracle(capture,'positive.audio_target_seq_idx').astype(np.int64)
    if x.shape[1]!=5376 or y.shape!=(len(x),32) or len(x)>100000 or indices.min()<0 or indices.max()>=len(x):raise ValueError('invalid head capture')
    selected=a.out/'selected.f32';x[indices].tofile(selected)
    for f in a.model.glob('*.safetensors'):
        with f.open('rb') as stream:
            n=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(n))
        if 'final_layer.audio_out.weight' in header:break
    else:raise ValueError('missing audio-head weights')
    offsets=[8+n+header['final_layer.audio_out.'+k]['data_offsets'][0] for k in ('weight','bias')]
    env=os.environ.copy();env['H3_SGLANG_CUBLAS_LIBRARY']=a.cublas;results={}
    for name,mode,input_path,rows in [('selected','linear-f32-bias',selected,len(indices)),
                                     ('full','linear-f32-bias',capture/(prefix+'input.0.bin'),len(x)),
                                     ('heuristic','linear-f32-head',selected,len(indices))]:
        output=a.out/(name+'-output.f32')
        command=[str(a.source.resolve()/'bin/cuda_sglang_replay'),mode,str(input_path),str(f),
                 *map(str,offsets),str(output),str(rows),'5376','32']
        if name=='heuristic':command.append(str(len(x)))
        run=subprocess.run(command,cwd=a.source,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True)
        (a.out/(name+'.log')).write_text(run.stdout+run.stderr);run.check_returncode()
        z=np.fromfile(output,dtype='<f4').reshape(rows,32)
        results[name]=dict(command=command,**metric(z[indices] if name=='full' else z,y[indices]))
    (a.out/'metrics.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))

if __name__=='__main__':main()
