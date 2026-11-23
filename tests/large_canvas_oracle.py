"""Independent pinned full-context BF16 network comparison at the new canvas."""
import argparse
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from large_canvas_guard import plan, run_case


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'target', 'weights', 'upstream', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--worker', action='store_true')
    a = ap.parse_args()
    contract = json.loads((Path(__file__).parent/'upscale/contract.json').read_text())
    assert sha(a.weights) == contract['artifact_sha256']
    assert sha(a.upstream) == contract['code_sha256']
    with a.source.open('rb') as f:
        header = f.read(160)
    w,h,frames,t,lh,lw = struct.unpack_from('<6I',header,24)
    p = plan('upscale',w*2,h*2,frames,refine=0)
    assert t in (7,12) and lh*lw <= 8160
    if not a.worker:
        with open('/tmp/h3-large-canvas-gpu.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # cuDNN oracle has additional full-volume intermediates/workspace.
            p['live_bytes'] = max(p['live_bytes'],24 << 30)
            p['capacity_bytes'] = (p['live_bytes']*10+8)//9
            run_case([sys.executable,*sys.argv,'--worker'],p,a.out.with_suffix('.oracle'),media=False)
        return
    import numpy as np
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.7)
    def av(path):
        data=path.read_bytes()
        assert data[:8] == b'H3AV\r\n\x1a\n'
        assert hashlib.sha256(data[:128]+data[160:]).digest() == data[128:160]
        width,height,fr,time,hh,ww = struct.unpack_from('<6I',data,24)
        vb,ab = struct.unpack_from('<2Q',data,72)
        assert len(data)==160+vb+ab and vb==24*time*hh*ww*4 and fr==frames
        return np.frombuffer(data[160:160+vb],dtype='<f4').copy().reshape(1,24,time,hh,ww),sha(path),hashlib.sha256(data[160+vb:]).hexdigest()
    source,source_hash,source_audio = av(a.source)
    target,target_hash,target_audio = av(a.target)
    assert source_audio==target_audio and target.shape==(1,24,t,lh*2,lw*2)
    names={'normalization','zero_module','ResBlockEmb3D','TemporalConv','LatentResizer3D'}
    selected=[n for n in ast.parse(a.upstream.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(selected)==len(names)
    scope=dict(torch=torch,nn=torch.nn,F=torch.nn.functional)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(a.upstream),'exec'),scope)
    net=scope['LatentResizer3D']().eval().to(device='cuda',dtype=torch.bfloat16)
    net.load_state_dict(load_file(str(a.weights),device='cpu'),strict=True)
    m=torch.tensor(contract['normalization']['mean'],device='cuda',dtype=torch.bfloat16).reshape(1,24,1,1,1)
    s=torch.tensor(contract['normalization']['std'],device='cuda',dtype=torch.bfloat16).reshape(1,24,1,1,1)
    with torch.inference_mode():
        x=torch.from_numpy(source).to(device='cuda',dtype=torch.bfloat16)
        y=net((x-m)/s,scale=2.,target_size=(t,lh*2,lw*2),enable_chunking=False)
        reference=(y*s+m).float().cpu().numpy()
    assert np.isfinite(reference).all() and np.isfinite(target).all()
    rms=max(float(np.sqrt(np.mean(reference.astype(np.float64)**2))),contract['tolerances']['rms_floor'])
    nrmse=float(np.sqrt(np.mean((reference.astype(np.float64)-target)**2)))/rms
    maximum=float(np.max(np.abs(reference-target)))/rms
    tolerance=contract['tolerances']['bf16_network']
    result=dict(passed=nrmse<=tolerance['nrmse'] and maximum<=tolerance['max_rms_error'],
        nrmse=nrmse,max_rms_error=maximum,tolerance=tolerance,source_sha256=source_hash,
        target_sha256=target_hash,audio_sha256=source_audio,shape=list(target.shape),
        artifact_sha256=contract['artifact_sha256'],upstream_sha256=contract['code_sha256'],
        torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        peak_torch_bytes=torch.cuda.max_memory_allocated(),full_context=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert result['passed']


if __name__=='__main__':
    main()
