#!/usr/bin/env python3
"""Gate a complete native render against the controlled official generation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from safetensors import safe_open
from refvideo_generation_oracle import snapshot
from test_refvideo_dit import compare


def decode_rgb(path):
    metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(path)]))['streams'][0]
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'])
    return np.frombuffer(raw,np.uint8).reshape(-1,metadata['height'],metadata['width'],3).astype(np.float32)/255


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-checkpoint',type=Path,required=True);p.add_argument('--native-video',type=Path,required=True)
    p.add_argument('--native-prefix',type=Path,help='also compare first-forward velocity captures')
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();parts,native,_,_=snapshot(args.native_checkpoint)
    import struct
    total,done=struct.unpack_from('<2i',parts[1],4);assert total==done and total==20
    # Fixed before real-media runs, same transformer bounds as the small test.
    limits={'relative_max':.15,'relative_l2':.10};results={}
    with safe_open(args.oracle/'generation.safetensors',framework='np') as f:
        for name in ('video','audio'):
            result=compare(native['x.'+name].numpy(),f.get_tensor('x.'+name+'_final'))
            result['passed']=all(result[k]<=v for k,v in limits.items());results[name]=result
            if args.native_prefix:
                want=f.get_tensor('x.'+name+'_velocity')
                got=np.fromfile(str(args.native_prefix)+'.'+name+'_velocity',np.float32).reshape(want.shape)
                result=compare(got,want);result['passed']=all(result[k]<=v for k,v in limits.items())
                results[name+'_velocity']=result
    # Compare identically compressed outputs; latent parity remains the stricter
    # numerical gate, while pixel checks catch decode/crop/timing regressions.
    got=decode_rgb(args.native_video);want=decode_rgb(args.oracle/'oracle.mp4')
    assert got.shape==want.shape
    diff=got-want;mae=float(np.abs(diff).mean());psnr=float(-10*np.log10(max(float((diff*diff).mean()),1e-30)))
    results['pixels']={'shape':list(got.shape),'mae':mae,'psnr_db':psnr,'passed':mae<=.05 and psnr>=20}
    result={'passed':all(x['passed'] for x in results.values()),'latent_limits':limits,
        'pixel_limits':{'mae':.05,'psnr_db_min':20},'cases':results,
        'native_checkpoint_sha256':hashlib.sha256(args.native_checkpoint.read_bytes()).hexdigest(),
        'native_video_sha256':hashlib.sha256(args.native_video.read_bytes()).hexdigest(),
        'oracle_preparation':json.loads((args.oracle/'preparation.json').read_text()),
        'oracle_generation':json.loads((args.oracle/'generation.json').read_text())}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(results,indent=2),flush=True)
    assert result['passed'],'complete-generation parity gate failed'
if __name__=='__main__':main()
