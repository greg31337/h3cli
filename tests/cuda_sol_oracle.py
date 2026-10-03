#!/usr/bin/env python3
"""Independent NumPy recipe-1 oracle; no CUDA arithmetic shared with candidate."""
import argparse,json,os,subprocess,time,hashlib
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bf(x):
    b=np.asarray(x,dtype='<f4').view('<u4');return ((b+0x7fff+((b>>16)&1))>>16).astype('<u2')
def fp(x):return (np.asarray(x,dtype='<u2').astype('<u4')<<16).view('<f4')
def metrics(x,y):
    x=x.astype('float64').ravel();y=y.astype('float64').ravel();d=x-y
    return dict(finite=bool(np.isfinite(x).all() and np.isfinite(y).all()),relative_l2=float(np.linalg.norm(d)/max(np.linalg.norm(y),1e-30)),max_relative=float(np.max(abs(d))/max(np.max(abs(y)),1e-30)),max_abs=float(np.max(abs(d))),cosine=float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-30)))
def metadata(seq,qb,protected=False,all_protected=False):
    result=[]
    for block in (qb,64):
        blocks=[]
        for start in range(0,seq,block):
            end=min(seq,start+block);protect=all_protected or (protected and (start<17 or end>seq-19 or start<=seq//2<end))
            blocks.append([int(protect),start//128 if protected else -1,(end-1)//128 if protected else -1,end-start])
        result.append(np.array(blocks,dtype='<i4'))
    return result

def write_layout(path,seq,meta):
    with Path(path).open('wb') as f:
        np.array([seq,len(meta[0]),len(meta[1])],dtype='<u4').tofile(f)
        for m in meta:m.tofile(f)
def read_layout(path):
    a=np.fromfile(path,dtype='<i4');n,q,k=a[:3];return int(n),[a[3:3+q*4].reshape(q,4),a[3+q*4:].reshape(k,4)]
def summaries(x,b):return fp(bf(np.array([x[i:i+b].mean(axis=0,dtype=np.float32) for i in range(0,len(x),b)])))
def routes(qc,kc,meta,scale,minimum,tau,radius,exact=False):
    qm,km=meta;count=len(km);mu=kc.mean(axis=0,dtype=np.float64);var=np.maximum((kc.astype('float64')**2).mean(axis=0)-mu*mu,0)
    a=scale*np.log2(np.e);threshold=qc@mu*a+tau*np.sqrt(((qc.astype('float64')**2)@var)*a*a+1e-6)
    dots=qc.astype('float64')@kc.astype('float64').T*a
    route=dots>threshold[:,None]
    route|=~np.isfinite(dots)|~np.isfinite(threshold[:,None]);route|=bool(exact)
    route|=(qm[:,0]!=0)[:,None]|(km[:,0]!=0)[None,:]
    local=(qm[:,1,None]>=0)&(km[None,:,1]>=0)&(qm[:,1,None]<=km[None,:,2]+radius)&(km[None,:,1]<=qm[:,2,None]+radius)
    floor=int(np.ceil(float(np.float32(minimum))*count));idx=np.arange(count,dtype='int64')
    route|=local|(((idx+1)*floor//count)!=(idx*floor//count))[None,:]|(km[:,3]<64)[None,:]
    return route

def reference(q,k,v,meta,qb,minimum,tau,radius,exact=False,indices=None):
    seq,heads,_=q.shape;idx=np.arange(seq) if indices is None else np.array(indices);out=np.empty((len(idx),heads,128),np.float64)
    masks=[];scale=float(fp(bf(np.array(1/np.sqrt(128),dtype='f4'))))
    qc=summaries(q,qb);kc=summaries(k,64);vc=summaries(v,64)
    for h in range(heads):
        route=routes(qc[:,h],kc[:,h],meta,scale,minimum,tau,radius,exact);masks.append(route)
        for b in np.unique(idx//qb):
            at=np.nonzero(idx//qb==b)[0];rows=idx[at];exact_rows=np.repeat(route[b],64)[:seq]
            keys=np.concatenate((k[exact_rows,h],kc[~route[b],h])).astype('float64')
            values=np.concatenate((v[exact_rows,h],vc[~route[b],h])).astype('float64')
            scores=q[rows,h].astype('float64')@keys.T*scale
            scores[:,int(exact_rows.sum()):]+=np.log(64.)
            scores-=scores.max(axis=1,keepdims=True);prob=np.exp(scores);prob/=prob.sum(axis=1,keepdims=True)
            out[at,h]=prob@values
    return out,np.array(masks,dtype='uint8')

def run(a,case,input_file=None,layout_file=None):
    dest=a.output/case['id'];dest.mkdir(parents=True,exist_ok=False)
    seq=case['sequence'];heads=case.get('heads',2);qb=case.get('q',32);minimum=case.get('minimum',.5);tau=case.get('tau',1);radius=case.get('radius',1)
    if input_file:
        raw=np.memmap(input_file,dtype='<u2',mode='r',shape=(3,seq,heads,128));q,k,v=[fp(x) for x in raw]
        _,meta=read_layout(layout_file)
    else:
        rng=np.random.default_rng(case.get('seed',1701));x=rng.normal(0,.5,(3,seq,heads,128)).astype('float32')
        pattern=case.get('pattern','random')
        if pattern=='zero':x[:]=0
        elif pattern=='constant':x[1]=.2;x[2]=.7
        elif pattern=='extreme':x[:2]*=40
        elif pattern=='spike':x[1,seq//2]*=32
        elif pattern=='nan':x[0,-1,-1,0]=np.nan
        elif pattern=='overflow':x[0]=np.float32(3e38)
        raw=bf(x);q,k,v=fp(raw);input_file=dest/'input.qkv';raw.tofile(input_file)
        meta=metadata(seq,qb,case.get('protected',False),case.get('all_protected',False));layout_file=dest/'layout.bin';write_layout(layout_file,seq,meta)
    command=[str(a.binary),'--input',str(input_file),'--layout',str(layout_file),'--output',str(dest/'out.bf16'),'--routes',str(dest/'routes.bin'),'--stats',str(dest/'native.json'),'--sequence',str(seq),'--heads',str(heads),'--q',str(qb),'--minimum',str(minimum),'--tau',str(tau),'--radius',str(radius),'--exact',str(int(case.get('exact',False))),'--head-major',str(int(case.get('head_major',False))),'--repeat',str(case.get('repeat',1)),'--failure-tests','1']
    start=time.time();p=subprocess.run(command,capture_output=True,text=True,timeout=case.get('timeout',180));(dest/'stderr.log').write_text(p.stderr)
    native=json.loads((dest/'native.json').read_text());row=dict(case=case,argv=command,returncode=p.returncode,native=native,input_sha256=sha(input_file),layout_sha256=sha(layout_file),binary_sha256=sha(a.binary))
    guards=native['guards'] and native['inputs_unchanged']
    if case.get('pattern') in ('nan','overflow'):passed=p.returncode==3 and guards and native['output_unchanged']
    else:
        out=fp(np.fromfile(dest/'out.bf16',dtype='<u2'))
        out=out.reshape(heads,seq,128).transpose(1,0,2) if case.get('head_major') else out.reshape(seq,heads,128)
        indices=None if seq<=1025 else sorted(set([0,1,31,32,seq//3,seq//2,seq-2,seq-1]))
        ref,masks=reference(q,k,v,meta,qb,minimum,tau,radius,case.get('exact',False),indices)
        got=out if indices is None else out[indices];row['oracle']=metrics(got,ref);row['sampled_query_rows']=indices
        row['route_match']=True
        if native['routes_exported']:row['route_match']=bool(np.array_equal(np.fromfile(dest/'routes.bin',dtype='uint8').reshape(masks.shape),masks))
        protected_rows=np.where(meta[0][:,0][np.arange(seq)//qb]!=0)[0]
        if len(protected_rows):
            protected_rows=protected_rows[np.linspace(0,len(protected_rows)-1,min(16,len(protected_rows)),dtype=int)]
            dense,_=reference(q,k,v,meta,qb,1,tau,radius,True,protected_rows)
            row['protected_dense']=metrics(out[protected_rows],dense)
        m=row['oracle'];passed=p.returncode==0 and guards and row['route_match'] and m['finite'] and m['relative_l2']<=.01 and m['max_relative']<=.02
        if not np.any(ref):passed=p.returncode==0 and guards and row['route_match'] and m['max_abs']<=1e-5
        if case.get('repeat',1)>1:passed &= native['free_after']>=native['free_before']
    row.update(passed=bool(passed),wall_seconds=time.time()-start);(dest/'result.json').write_text(json.dumps(row,indent=2)+'\n');print(case['id'],row['passed'],row.get('oracle'),flush=True);return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,default=Path('./bin/cuda_sol_native'));p.add_argument('--output',required=True,type=Path);p.add_argument('--input',type=Path);p.add_argument('--layout',type=Path);p.add_argument('--sequence',type=int);p.add_argument('--heads',type=int,default=56);p.add_argument('--minimum',type=float,default=.75);p.add_argument('--case');a=p.parse_args();a.binary=a.binary.resolve();a.output.mkdir(parents=True,exist_ok=True)
    if a.input:cases=[dict(id='real',sequence=a.sequence,heads=a.heads,minimum=a.minimum,repeat=6)]
    else:cases=json.loads(Path('tests/cuda_sol_acceptance.json').read_text())['operator_cases']
    if a.case:cases=[c for c in cases if c['id']==a.case]
    rows=[run(a,c,a.input,a.layout) for c in cases];(a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n');return 0 if all(r['passed'] for r in rows) else 1
if __name__=='__main__':raise SystemExit(main())
