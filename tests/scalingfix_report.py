#!/usr/bin/env python3
"""Collect numerical/performance evidence without claiming a visual quality win."""
import json,platform,statistics,subprocess,re
from pathlib import Path
from scalingfix_metrics import OUT,ROOT

def main():
    read=lambda p:json.loads(p.read_text())
    data={'baseline':'91546c572f8f6548ddd7da400442b0ffbf8eeae2','machine':platform.platform(),
        'kernels':[json.loads(s) for s in (OUT/'kernels.jsonl').read_text().splitlines()],
        'encoder_rows':read(OUT/'conditioning-row-metrics.json'),'encoder':read(OUT/'encoder-metrics.json'),'captured_attention':read(OUT/'captured-attention-metrics.json'),
        'text_weight_identity':read(OUT/'text-weight-identity.json'),'legacy_kernel_compatibility':read(OUT/'legacy-kernel-compatibility.json'),'encoder_performance':{}}
    for case in ('plain','dialogue','image','video'):
        data['encoder_performance'][case]={mode:read(OUT/'encoder'/case/mode/'result.json') for mode in ('legacy','scaled-q','reference')}
        rows=data['encoder'][case]
        for oracle in ('official-sdpa','official-fp32-eager'):
            for layer,stats in rows['reference vs '+oracle].items():
                assert stats['relative_l2']<.02 and stats['cosine']>.9998,(case,oracle,layer,stats)
        assert all(x['bf16_mismatch']==0 for x in rows['reference-repeat vs reference'].values()),case
        att=data['captured_attention'][case];assert att['reference']['rmse'] < att['legacy-synchronized']['rmse']*.1
        assert att['reference']['rmse'] <= att['scaled-q']['rmse']*1.1
    if (OUT/'benchmark.jsonl').exists():
        bench=[json.loads(s) for s in (OUT/'benchmark.jsonl').read_text().splitlines()]
        data['benchmark']={str(n):{m:statistics.median(r['gpu_seconds'] for r in bench if r['sequence']==n and r['mode']==m) for m in ('legacy','scaled-q','reference')} for n in sorted({r['sequence'] for r in bench})}
    for key,file in [('generation','generation/results.json'),('media','generation/media-metrics.json'),('speech','generation/speech.json'),('review','generation/review.json'),('default_regression','default/results.json')]:
        if (OUT/file).exists():data[key]=read(OUT/file)
    if 'generation' in data:
        for case,record in data['generation'].items():
            log=OUT/'generation'/f'{case}.log'
            if log.exists():
                text=log.read_text();phases={}
                for label,pattern in [('qwen',r'h3(?:cli)? profile: Qwen text encoder\s+total\s+wall=\s*([0-9.]+)s'),('denoise',r'h3(?:cli)? profile: H3 DiT\s+Euler denoise\s+wall=\s*([0-9.]+)s')]:
                    match=re.search(pattern,text)
                    if match:phases[label]=float(match[1])
                record['phase_seconds']=phases
    (ROOT/'docs/scalingfix-validation.json').write_text(json.dumps(data,indent=2)+'\n')
    print('PASS independent attention, <2% full-encoder parity gate, exact reference repeats; report collected')
if __name__=='__main__':main()
