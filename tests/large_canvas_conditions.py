"""Compare exported raw reference retargeting against the frozen independent network."""
import argparse
import ast
import json
from pathlib import Path
from upscale_reference import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('weights','upstream','directory'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    contract=json.loads((Path(__file__).parent/'upscale/contract.json').read_text())
    assert sha(a.weights)==contract['artifact_sha256'] and sha(a.upstream)==contract['code_sha256']
    data=json.loads((a.directory/'conditions.json').read_text())
    for c in data['cases']:
        for shape in (c['input'],c['target']):
            assert shape[:2]==[1,24] and shape[2] in (1,7,12) and 0<min(shape[3:])<=max(shape[3:])<=240
    import numpy as np
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.7)
    names={'normalization','zero_module','ResBlockEmb3D','TemporalConv','LatentResizer3D'}
    nodes=[n for n in ast.parse(a.upstream.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(nodes)==len(names)
    scope=dict(torch=torch,nn=torch.nn,F=torch.nn.functional)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(a.upstream),'exec'),scope)
    net=scope['LatentResizer3D']().eval().to(device='cuda',dtype=torch.bfloat16)
    net.load_state_dict(load_file(str(a.weights)),strict=True)
    mean=torch.tensor(contract['normalization']['mean'],device='cuda',dtype=torch.bfloat16).reshape(1,24,1,1,1)
    std=torch.tensor(contract['normalization']['std'],device='cuda',dtype=torch.bfloat16).reshape(1,24,1,1,1)
    records=[]
    for c in data['cases']:
        x=np.fromfile(a.directory/f"{c['id']}-input.f32",dtype='<f4').reshape(c['input'])
        target=np.fromfile(a.directory/f"{c['id']}-target.f32",dtype='<f4').reshape(c['target'])
        if not data['anchors'] and data['image_size'] in (1,2):
            assert c['input']==c['target'], 'high/max must preserve intrinsic reference size'
        if c['input']==c['target']:
            assert np.array_equal(x,target)
            records.append(dict(id=c['id'],passed=True,identity=True))
            continue
        with torch.inference_mode():
            native=torch.from_numpy(x).to(device='cuda',dtype=torch.bfloat16)
            # Pinned upstream TARGET_DIMENSIONS mode derives the embedding
            # from both spatial ratios; a retargeted reference need not double.
            scale=(c['target'][3]/c['input'][3]+c['target'][4]/c['input'][4])/2
            y=net((native-mean)/std,scale=scale,target_size=tuple(c['target'][2:]),enable_chunking=False)
            reference=(y*std+mean).float().cpu().numpy()
        assert np.isfinite(target).all()
        rms=max(float(np.sqrt(np.mean(reference.astype(np.float64)**2))),contract['tolerances']['rms_floor'])
        nrmse=float(np.sqrt(np.mean((reference.astype(np.float64)-target)**2)))/rms
        maximum=float(np.max(np.abs(reference-target)))/rms
        tol=contract['tolerances']['bf16_network']
        records.append(dict(id=c['id'],passed=nrmse<=tol['nrmse'] and maximum<=tol['max_rms_error'],nrmse=nrmse,max_rms_error=maximum))
    result=dict(passed=bool(records) and all(r['passed'] for r in records),cases=records,
        full_context=True,contract_sha256=sha(Path(__file__).parent/'upscale/contract.json'))
    (a.directory/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert result['passed']


if __name__=='__main__':
    main()
