#!/usr/bin/env python3
"""Make bounded FP32 convolution fixtures from retained encoder operations."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from cuda_sglang_compare import oracle

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('model',type=Path);p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    from safetensors import safe_open
    records={}
    with safe_open(a.model/'model.safetensors',framework='np') as f:
        for name in ('encoder.conv_in','encoder.down.0.block.0.conv1','encoder.down.2.downsample.conv'):
            out=a.out/name;out.mkdir()
            down=name.endswith('downsample.conv')
            x=oracle(a.capture,'encoder.down.2.block.1.output' if down else name+'.input')
            y=oracle(a.capture,'encoder.down.2.downsample.output' if down else name+'.output')
            # Both audited BaseConv3d modules reflect the spatial edge and
            # prepend two causal zero planes, before a valid 3D convolution.
            edge=(0,1) if down else (1,1)
            x=np.pad(x,((0,0),(0,0),(0,0),edge,edge),mode='reflect')
            x=np.pad(x,((0,0),(0,0),(2,0),(0,0),(0,0)))
            tensors=dict(input=x,output=y,weight=f.get_tensor(name+'.weight'),bias=f.get_tensor(name+'.bias'))
            records[name]={}
            for label,t in tensors.items():
                raw=t.astype('<f4',copy=False).tobytes();(out/(label+'.f32')).write_bytes(raw)
                records[name][label]=dict(shape=list(t.shape),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    (a.out/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
