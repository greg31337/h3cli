#!/usr/bin/env python3
"""Independent FP32 attention on captured first-layer native Q/K/V.

All modes enter attention with bit-identical data. This separates attention
scaling from pre-existing official/native norm, RoPE and activation boundaries.
"""
import json
from pathlib import Path
import numpy as np
from scalingfix_metrics import read,compare,OUT

def bfround(x):
    u=x.astype(np.float32).view(np.uint32);return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
def main():
    record={}
    for case in ('plain','dialogue','image','video'):
        d=OUT/'encoder'/case
        q=read(d/'reference/first-query.bf16').reshape(-1,64,128);n=len(q)
        k=read(d/'reference/first-key.bf16').reshape(n,8,128);v=read(d/'reference/first-value.bf16').reshape(n,8,128)
        for mode in ('legacy','scaled-q','legacy-synchronized'):
            for name in ('query','key','value'):assert (d/mode/f'first-{name}.bf16').read_bytes()==(d/'reference'/f'first-{name}.bf16').read_bytes()
        out=np.empty_like(q);mask=np.triu(np.ones((n,n),bool),1)
        for h in range(64):
            score=(q[:,h]@k[:,h//8].T)*np.float32(1/np.sqrt(128));score[mask]=-np.inf
            score=np.exp(score-score.max(axis=1,keepdims=True));score/=score.sum(axis=1,keepdims=True);out[:,h]=score@v[:,h//8]
        rounded=bfround(out).reshape(-1)
        record[case]={m:compare(read(d/m/'first-attention.bf16'),rounded) for m in ('legacy','scaled-q','reference','legacy-synchronized')}
        print(case,record[case],flush=True)
    (OUT/'captured-attention-metrics.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
