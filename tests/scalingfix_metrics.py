#!/usr/bin/env python3
"""Encoder statistics in float64, including exact stored BF16 mismatch rates."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/scalingfix-validation'
def read(p):return (np.fromfile(p,'<u2').astype(np.uint32)<<16).view(np.float32)
def compare(a,b):
    a=a.astype(np.float64);b=b.astype(np.float64);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    d=a-b;norm=np.linalg.norm(b);return {'max_abs':float(np.abs(d).max()),'mean_abs':float(np.abs(d).mean()),'rmse':float(np.sqrt(np.mean(d*d))),
        'relative_l2':float(np.linalg.norm(d)/max(norm,1e-30)),'cosine':float(np.dot(a,b)/max(np.linalg.norm(a)*norm,1e-30)), 'bf16_mismatch':float(np.mean(a!=b))}
def row_metrics():
    results={}
    for case in ('plain','dialogue','image','video'):
        d=OUT/'encoder'/case;fixture=OUT/'presentations'/case
        if not (d/'official-sdpa/layer-50.bf16').exists():continue
        ref=read(d/'official-sdpa/layer-50.bf16').reshape(-1,5120)
        tags=np.fromfile(fixture/'tags.u8','u1') if (fixture/'tags.u8').exists() else np.ones(len(ref),np.uint8)
        result={}
        for mode in ('legacy','scaled-q','reference'):
            actual=read(d/mode/'layer-50.bf16').reshape(-1,5120)
            rows=np.linalg.norm((actual-ref).astype(np.float64),axis=1)/np.maximum(np.linalg.norm(ref.astype(np.float64),axis=1),1e-20)
            loc=np.unravel_index(np.argmax(abs(actual-ref)),actual.shape)
            result[mode]={'language_rows':compare(actual[tags==1].reshape(-1),ref[tags==1].reshape(-1)),
                'last_token':compare(actual[-1],ref[-1]),'row_relative_l2_median':float(np.median(rows)),
                'row_relative_l2_p95':float(np.quantile(rows,.95)),'largest_error_location':list(map(int,loc)),
                'largest_error_native':float(actual[loc]),'largest_error_official':float(ref[loc])}
        results[case]=result
    (OUT/'conditioning-row-metrics.json').write_text(json.dumps(results,indent=2)+'\n')

def main():
    row_metrics()
    record={}
    for case in ('plain','dialogue','image','video'):
        d=OUT/'encoder'/case;record[case]={}
        for left,right in [('legacy','baseline'),('reference-repeat','reference'),('legacy-synchronized','official-sdpa'),('legacy','official-sdpa'),('scaled-q','official-sdpa'),('reference','official-sdpa'),('legacy','official-fp32-eager'),('scaled-q','official-fp32-eager'),('reference','official-fp32-eager'),('legacy','reference'),('scaled-q','reference')]:
            result={}
            for layer in [0,1,2,3,4,5,10,25,40,50]:
                name=f'layer-{layer:02d}.bf16';a=d/left/name;b=d/right/name
                if a.exists() and b.exists():result[str(layer)]=compare(read(a),read(b))
            if result:record[case][left+' vs '+right]=result
    (OUT/'encoder-metrics.json').write_text(json.dumps(record,indent=2)+'\n')
    for case,rows in record.items():
        for pair,layers in rows.items():
            if '50' in layers:print(case,pair,layers['50'])
if __name__=='__main__':main()
