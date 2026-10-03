#!/usr/bin/env python3
"""CPU-only native byte/PCM checks against pinned Pillow and torchaudio."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from cuda_sglang_compare import metric


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','fixtures','jpeg','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--delivery-ffmpeg',type=Path,help='Also prove the output encoder override cannot change reference audio')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    env=os.environ|dict(H3_SGLANG_JPEG_LIBRARY=str(a.jpeg));rows=[]
    for name,target in [('portrait.jpg',(2048,2720)),('portrait.jpg',(1365,1821)),('portrait.jpg',(320,416)),('first.png',(320,240)),('first.png',(672,512))]:
        image=Image.open(a.fixtures/name).convert('RGB');dest=a.out/f'{Path(name).stem}-{target[0]}x{target[1]}.f32'
        cmd=[str(a.source/'bin/cuda_sglang_media_input'),str(a.fixtures/name),*map(str,image.size),*map(str,target),str(dest)]
        result=subprocess.run(cmd,stdin=subprocess.DEVNULL,capture_output=True,text=True,env=env)
        (dest.with_suffix('.log')).write_text(result.stdout+result.stderr);result.check_returncode()
        native=np.fromfile(dest,'<f4').reshape(3,target[1],target[0]).transpose(1,2,0)
        n=np.rint(native*255).astype(np.uint8);o=np.asarray(image.resize(target,Image.Resampling.LANCZOS))
        m=metric(n,o);rows.append(dict(input=name,target=target,command=cmd,metric=m,passed=bool(m.get('exact'))))
        (a.out/'images.json').write_text(json.dumps(dict(results=rows,passed=all(r['passed'] for r in rows)),indent=2)+'\n')
    # Use the installed helper to retain precisely the decoder and resampler
    # chosen by the pinned reference-video material chain, with CPU tensors.
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import _load_waveform,_audio_resampler
    import torch
    raw,rate=_load_waveform(str(a.fixtures/'reference.mp4'),material_chain='video.reference_preserve',max_duration_seconds=124/24)
    with torch.inference_mode():expected=_audio_resampler(rate)(raw).numpy()
    dest=a.out/'soundtrack.f32';cmd=[str(a.source/'bin/cuda_sglang_soundtrack_input'),str(a.fixtures/'reference.mp4'),str(dest)]
    result=subprocess.run(cmd,stdin=subprocess.DEVNULL,capture_output=True,text=True,env=env)
    (a.out/'soundtrack.log').write_text(result.stdout+result.stderr);result.check_returncode()
    got=np.fromfile(dest,'<f4').reshape(2,-1);m=metric(got,expected)
    expected.astype('<f4').tofile(a.out/'oracle-soundtrack.f32')
    rows.append(dict(input='reference.mp4',command=cmd,metric=m,passed=bool(m.get('exact'))))
    if a.delivery_ffmpeg:
        override=a.out/'soundtrack-output-override.f32'
        cmd=[str(a.source/'bin/cuda_sglang_soundtrack_input'),str(a.fixtures/'reference.mp4'),str(override)]
        result=subprocess.run(cmd,stdin=subprocess.DEVNULL,capture_output=True,text=True,
                              env=env|{'H3_FFMPEG':str(a.delivery_ffmpeg.resolve(strict=True))})
        (a.out/'soundtrack-output-override.log').write_text(result.stdout+result.stderr);result.check_returncode()
        m=metric(np.fromfile(override,'<f4').reshape(2,-1),expected)
        rows.append(dict(input='reference.mp4',output_ffmpeg=str(a.delivery_ffmpeg),command=cmd,
                         metric=m,passed=bool(m.get('exact'))))
    report=dict(kind='native preprocessing vs pinned CPU oracle; not final conditioning qualification',results=rows,passed=all(r['passed'] for r in rows))
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
