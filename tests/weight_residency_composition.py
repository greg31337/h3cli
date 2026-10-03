#!/usr/bin/env python3
"""Bounded paused-state placement comparisons; no quantized video campaign."""
import argparse,json,os,re,subprocess,time
from pathlib import Path
from cuda_weight_residency import sha,write,PROMPT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','binary','model','out','baseline-av'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--quant-cache',type=Path,required=True)
    p.add_argument('--packed-capacity',type=int,required=True,help='Test-only total byte cap which must force packed streaming')
    p.add_argument('--group',choices=('composition','quant'),required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source=a.source.resolve();fixtures=source/'tests/fixtures/cuda-reference'
    cases=[('fl2va',['--first-frame',str(fixtures/'first.png'),'--last-frame',str(fixtures/'portrait.jpg')],2,6),
        ('ordered-ref2va',['--ref-image',str(fixtures/'portrait.jpg'),'--ref-image',str(fixtures/'first.png')],2,6),
        ('continuation',['--continue-from',str(a.baseline_av.resolve())],2,6),
        ('reuse',['--reuse','2'],6,50),('core-reuse',['--core-reuse','4'],6,50),
        ('adaptive',['--adaptive-cache','conservative','--adaptive-cache-warmup','2'],6,50),
        ('subblock',['--cuda-attention','subblock','--subblock-sparsity','0.75','--subblock-warmup','2'],6,50),
        ('adaptive-subblock',['--adaptive-cache','conservative','--adaptive-cache-warmup','2','--cuda-attention','subblock','--subblock-sparsity','0.75','--subblock-warmup','2'],6,50)]
    if a.group=='quant':cases=[(q,['--cuda-denoise-quant',q,'--cuda-denoise-quant-cache',str(a.quant_cache.resolve())],2,6) for q in ('fp8','nvfp4')]
    result={'passed':False,'runs':[]};write(out/'result.json',result)
    for name,flags,stop,steps in cases:
        gold=None;decisions=None
        placements=[('stream',None),('auto',2),('auto',None)] if a.group=='composition' else [('resident',None),('stream',None),('auto',None),('auto',a.packed_capacity)]
        for i,(mode,cap) in enumerate(placements):
            d=out/('%s-%s-%d'%(name,mode,i));d.mkdir();(d/'steps').mkdir();(d/'preparation').mkdir()
            e=dict(os.environ);e.update(H3_TEST_MAX_EVALUATIONS='6',H3_CUDA_WEIGHT_MODE=mode,H3_TEST_NATIVE_STEP_DIR=str(d/'steps'),H3_TEST_SGLANG_DIR=str(d/'preparation'),H3_SGLANG_CAPTURE_STEPS='none',H3_EXPERIMENT_TRACE='1')
            for k in ['H3_TEST_CUDA_RESIDENT_BLOCKS','H3_TEST_CUDA_CAPACITY_BYTES','H3_CUDA_TEST_MEMORY_BUDGET']:e.pop(k,None)
            if cap is not None:e['H3_TEST_CUDA_RESIDENT_BLOCKS' if a.group=='composition' else 'H3_TEST_CUDA_CAPACITY_BYTES']=str(cap)
            w,h,frames=(640,480,56) if name=='continuation' else (256,256,22)
            cmd=[str(a.binary.resolve()),'-d',str(a.model.resolve()),'-p',PROMPT,'--width',str(w),'--height',str(h),'--frames',str(frames),'--steps',str(steps),'--seed','42','--stop-after-step',str(stop),'--save-sampler-state',str(d/'paused.h3sample'),*flags]
            record={'argv':cmd,'mode':mode,'cap':cap,'passed':False};write(d/'result.json',record);start=time.monotonic()
            with (d/'run.log').open('wb') as f:r=subprocess.run(cmd,cwd=source,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=900)
            record.update(returncode=r.returncode,seconds=time.monotonic()-start);write(d/'result.json',record);assert r.returncode==0,d
            record['numeric']={str(p.relative_to(d)):sha(p) for sub in ('steps','preparation') for p in sorted((d/sub).rglob('*')) if p.is_file()}
            assert record['numeric'] and (d/'paused.h3sample').stat().st_size>1024
            if gold is None:gold=record['numeric']
            else:assert record['numeric']==gold,(name,mode,cap,'numeric mismatch')
            log=(d/'run.log').read_text();record['plans']=re.findall(r'h3cli: CUDA (?:BF16|packed) weight planner: ([^\n]+)',log)
            record['adaptive']=re.findall(r'adaptive step=(\d+).*?decision=(\S+) reason=(\S+).*?blocks=(\d+)',log)
            if decisions is None:decisions=record['adaptive']
            else:assert record['adaptive']==decisions,(name,'changed adaptive decisions')
            if a.group=='composition' and cap is not None:assert any('effective=partial resident=2/50 ' in q for q in record['plans'])
            if a.group=='quant' and mode=='auto':assert any(('effective=stream' if cap else 'effective=resident') in q for q in record['plans'])
            record['passed']=True;write(d/'result.json',record);result['runs'].append({'id':d.name,'passed':True});write(out/'result.json',result)
            print(d.name,record['seconds'],record['plans'],record['adaptive'],flush=True)
    result['passed']=True;write(out/'result.json',result)
if __name__=='__main__':main()
