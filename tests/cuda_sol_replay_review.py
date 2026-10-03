#!/usr/bin/env python3
"""Compare real routing windows and dense/forced-exact outputs on retained inputs."""
import fcntl,json,re,statistics,sys
from pathlib import Path
import numpy as np
from cuda_sol_campaign import ROOT,execute,verify,save
from cuda_sol_oracle import fp,bf,read_layout,summaries,routes,metrics,reference,sha

def main():
    with (ROOT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify();minimum=json.loads((ROOT/'candidate.json').read_text())['minimum'];results=[]
        for i,path in enumerate(sorted((ROOT/'diagnostics/C-calibration-default/qkv').glob('*.qkv'))):
            directory=ROOT/'real-replay'/f'block-{i}';mask=directory/'last-window.mask';stats=directory/'last-window.json'
            execute('C-route-window-'+str(i),'C',['./bin/cuda_sol_route_probe',str(path),str(path.parent/'layout.bin'),str(minimum),str(mask),str(stats)],90,allow_failure=False)
            s=json.loads(stats.read_text());seq=s['sequence'];_,meta=read_layout(path.parent/'layout.bin');raw=np.memmap(path,dtype='<u2',mode='r',shape=(3,seq,56,128))
            first=s['first_head'];stop=first+s['group'];qc=summaries(fp(raw[0,:,first:stop]),32);kc=summaries(fp(raw[1,:,first:stop]),64)
            scale=float(fp(bf(np.array(1/np.sqrt(128),dtype='f4'))));expected=np.array([routes(qc[:,h],kc[:,h],meta,scale,minimum,1,1)[s['first_query']:] for h in range(s['group'])],dtype='uint8')
            got=np.fromfile(mask,dtype='uint8').reshape(expected.shape);match=bool(np.array_equal(got,expected));protected=(meta[0][s['first_query']:,0]!=0)[None,:,None]|(meta[1][:,0]!=0)[None,None,:]
            protected_exact=bool(np.all(got[np.broadcast_to(protected,got.shape)]==1))
            exact=np.memmap(directory/'forced-exact.bf16',dtype='<u2',mode='r',shape=(seq,56,128));dense=np.memmap(directory/'dense.bf16',dtype='<u2',mode='r',shape=exact.shape)
            # Independently recompute dense SDPA at a fixed sparse row sample.
            indices=sorted(set([0,1,31,32,seq//3,seq//2,seq-2,seq-1]));q,k,v=[fp(a) for a in raw]
            ref,_=reference(q,k,v,meta,32,1,1,1,True,indices)
            em=metrics(fp(exact[indices]),ref);dm=metrics(fp(dense[indices]),ref)
            native_times=json.loads((directory/'real/native.json.timings.json').read_text())[1:];forced_times=json.loads((directory/'forced-native.json.timings.json').read_text())[1:]
            dense_perf=json.loads((ROOT/f'runs/C-dense-replay-{i}/stdout.log').read_text())
            row=dict(block=i,sequence=seq,mask_window=s,mask_sha256=sha(mask),mask_match=match,mask_mismatches=int(np.count_nonzero(got!=expected)),protected_pairs_exact=protected_exact,forced_exact_oracle=em,dense_oracle=dm,native_mean=statistics.mean(native_times),forced_mean=statistics.mean(forced_times),dense_mean=dense_perf['mean_seconds'],native_samples=native_times,forced_samples=forced_times,dense_samples=dense_perf['samples_seconds'])
            row.update(native_stdev=statistics.stdev(native_times),forced_stdev=statistics.stdev(forced_times),dense_stdev=statistics.stdev(dense_perf['samples_seconds']))
            row['passed']=match and protected_exact and em['finite'] and em['relative_l2']<=.01 and em['max_relative']<=.02;results.append(row);save(ROOT/'replay-review.json',results)
        assert all(r['passed'] for r in results),'Real routing/exact gate failed'
if __name__=='__main__':main()
