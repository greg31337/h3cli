#!/usr/bin/env python3
"""Fixed SubBlock 0.75 + FP8/NVFP4 triplets; no adaptive cache or repeated videos."""
import json
import re
import sys
import cuda_adaptive_quant as campaign
from cuda_reference_regression import PROMPT

CASES=[]
for group,precision in [('F8','fp8'),('F4','nvfp4')]:
    for suffix,args in [('D0',[]),('Q',['--cuda-denoise-quant',precision]),
                        ('S75',['--cuda-denoise-quant',precision,'--cuda-attention','subblock','--subblock-sparsity','0.75'])]:
        CASES.append(dict(id=f'{group}-{suffix}',group=precision,baseline=f'{group}-D0',args=args))
IDS=[v['id'] for v in CASES]


def manifest(path):
    value=json.loads(path.read_text())
    assert [value[k] for k in ('schema','recipe','width','height','frames','steps','fps','seed','timeout_seconds')]==[1,1,640,480,90,50,24,42,7200]
    assert value['prompt']==PROMPT and value['decoder']=='full' and value['weight_mode']=='resident'
    assert [value[k] for k in ('quantized_execution_recipe','subblock_execution_recipe','subblock_kernel_recipe','subblock_plan','packing_recipe','subblock_sparsity')]==[4,2,1,1,2,.75]
    assert value['variants']==CASES,'changed fixed SubBlock comparison'
    return value


def validate_dispatch(variant,steps,cache,log,presentation):
    quant='--cuda-denoise-quant' in variant['args'];sparse='subblock' in variant['args']
    assert not cache and [s['step'] for s in steps]==list(range(50))
    assert all(s['evaluated']==1 and s['blocks']==50 and s['quant_calls']==(200 if quant else 0) for s in steps)
    for s in steps:
        active=sparse and s['step']>=10
        assert s['sparse_calls']==(49 if active else 0) and s['dense_calls']==(1 if active else 50)
        # This fixed workload has 132 Q64 blocks and 56 heads. Plan 1 keeps
        # all heads together and tiles queries in slabs of 128: two routing
        # launches per sparse block (src/denoise/subblock.c, cuda_subblock.cu).
        assert s['router_calls']==(98 if active else 0), 'wrong tiled routing dispatch'
        if active:
            assert s['protected_calls']==98 and s['dense_bypass']==1
            assert s['possible']==49*56*132*132, 'changed fixed attention geometry'
            assert 0<s['selected']<s['possible']
        else:assert s['protected_calls']==0 and s['selected']==s['possible']==0
    if quant:
        precision=variant['group'];mode=1 if precision=='fp8' else 2;recipe=4 if sparse else 2
        assert f'DiT quantization={precision} recipe={recipe}' in log
        assert 'weights=compressed-resident' in log and 'weights=compressed-stream' not in log
        counters=re.findall(r'projection counters requested='+precision+r' recipe=(\d+) native_calls=(\d+) cache_hits=(\d+) prepared=(\d+)',log)
        assert counters==[(str(recipe),'10000','200','0')],counters
        assert re.search(r'^denoise_quant '+str(mode)+' '+str(recipe)+r' [0-9a-f]{64}$',presentation,re.M)
    else:assert not re.search(r'^denoise_quant [12] ',presentation,re.M)
    if sparse:
        assert 'DiT attention=subblock recipe=2 plan=1 sparsity=0.75 warmup=10 probe=dense' in log
        assert re.search(r'^attention 4 2 1$',presentation,re.M)
        assert re.search(r'^adaptive 0 0$',presentation,re.M)
        value=re.search(r'^subblock (\S+)$',presentation,re.M)
        assert value and float.fromhex(value[1])==.75
    assert 'CUDA weight planner: resident' in log and 'CUDA weight planner: stream' not in log
    return dict(quantized_calls=10000 if quant else 0,quantized_projection_matrices=200 if quant else 0,
                quantization_recipe=4 if sparse else 2 if quant else 0,adaptive_recipe=0,
                attention_recipe=2 if sparse else 0,subblock_sparsity=.75 if sparse else 0,weight_mode='resident')


if __name__=='__main__':campaign.main(comparison=sys.modules[__name__])
