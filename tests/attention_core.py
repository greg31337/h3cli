#!/usr/bin/env python3
"""Compare full-core BF16 hidden traces, including query-domain error."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
p.add_argument('--segments-log',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
a=p.parse_args();segments={}
for kind,start,stop in re.findall(r'attention validation segment kind=(\d+) start=(\d+) stop=(\d+)',a.segments_log.read_text()):
    name=['text','condition','reference_image','reference_audio','audio','video'][int(kind)]
    segments.setdefault(name,[]).extend(range(int(start),int(stop)))
def metrics(x,y):
    x=x.astype('float64').reshape(-1);y=y.astype('float64').reshape(-1);d=y-x
    xn=np.linalg.norm(x);yn=np.linalg.norm(y)
    return {'finite':bool(np.isfinite(x).all() and np.isfinite(y).all()),'relative_l2':float(np.linalg.norm(d)/max(xn,1e-30)),
        'max_abs':float(np.abs(d).max()),'cosine':float(x@y/max(xn*yn,1e-30))}
rows=[]
for path in sorted(a.reference.glob('step-*-block-*.bf16')):
    candidate=a.candidate/path.name
    def read(file):
        raw=np.fromfile(file,dtype='<u2')
        digest=hashlib.sha256(raw).hexdigest()
        return (raw.astype('uint32')<<16).view('float32').reshape(-1,5376),digest,raw.nbytes
    x,x_hash,nbytes=read(path);y,y_hash,_=read(candidate);assert x.shape==y.shape
    rows.append({'name':path.stem,'reference_sha256':x_hash,'candidate_sha256':y_hash,
        'bytes':nbytes,'all':metrics(x,y),
        'domains':{k:metrics(x[indices],y[indices]) for k,indices in segments.items()}})
if not rows:raise RuntimeError('no full-core traces')
a.output.write_text(json.dumps(rows,indent=2)+'\n')
