#!/usr/bin/env python3
"""Every-frame explicit-option diagnostics against the historical fast contract, using CPU metrics and unchanged frozen thresholds."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from cuda_sglang_compare import metric
from cuda_reference_regression import sha,write

def main():
 p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('out',type=Path);p.add_argument('--contract',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 c=json.loads(a.contract.read_text());g=c['gates'];meta=[];start=time.monotonic()
 for path in [a.reference,a.candidate]:
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(path)]));meta.append(probe)
 videos=[next(s for s in m['streams'] if s['codec_type']=='video') for m in meta]
 for k in ['width','height','nb_read_frames','avg_frame_rate']:assert videos[0][k]==videos[1][k]
 audios=[next(s for s in m['streams'] if s['codec_type']=='audio') for m in meta]
 for k in ['channels','sample_rate']:assert audios[0][k]==audios[1][k]
 w,h=videos[0]['width'],videos[0]['height'];count=int(videos[0]['nb_read_frames']);frames=[];net=None;previous=None;worst=2;dec=[]
 try:
  for path in [a.reference,a.candidate]:dec.append(subprocess.Popen(['ffmpeg','-v','error','-i',str(path),'-an','-pix_fmt','rgb24','-f','rawvideo','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE))
  for i in range(count):
   raw=[d.stdout.read(w*h*3) for d in dec];assert all(len(r)==w*h*3 for r in raw)
   x,y=[np.frombuffer(r,np.uint8).reshape(h,w,3).astype(np.float32)/255 for r in raw];delta=y-x
   if raw[0]==raw[1]:ssim=1.;lp=0.
   else:
    if net is None:
     import torch,lpips
     torch.set_num_threads(4);net=lpips.LPIPS(net='alex',version='0.1').eval()
    with torch.inference_mode():lp=float(net(*(torch.from_numpy(v).permute(2,0,1)[None]*2-1 for v in [x,y])).item())
    ssim=float(structural_similarity(x,y,data_range=1,channel_axis=-1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False))
   temporal=float(np.sqrt(np.square(delta-previous,dtype=np.float64).mean())) if previous is not None else 0.;previous=delta
   frames.append(dict(frame=i,ssim=ssim,lpips=lp,temporal_error_rms=temporal))
   if ssim<worst:
    worst=ssim
    for name,z in [('reference',x),('candidate',y),('difference-x20',np.minimum(abs(delta)*20,1))]:Image.fromarray(np.rint(z*255).astype(np.uint8)).save(a.out/(name+'-worst.png'))
  for d in dec:assert not d.stdout.read(1) and not d.stderr.read() and d.wait()==0
 finally:
  for d in dec:
   if d.poll() is None:d.terminate();d.wait()
 pcm=[]
 for path in [a.reference,a.candidate]:pcm.append(np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-vn','-acodec','pcm_f32le','-f','f32le','-']),dtype='<f4'))
 assert pcm[0].shape==pcm[1].shape
 audio=metric(pcm[1],pcm[0]);checks=dict(frames=count==len(frames),ssim=all(x['ssim']>=g['every_frame_ssim_min'] for x in frames),lpips=all(x['lpips']<=g['every_frame_lpips_alex_v01_max'] for x in frames),temporal=all(x['temporal_error_rms']<=g['temporal_error_rms_max'] for x in frames),audio=bool(audio['finite'] and audio['relative_l2']<=g['audio_relative_l2_max'] and audio['cosine']>=g['audio_cosine_min']),av_duration_samples=True)
 r=dict(passed=all(checks.values()),checks=checks,frames=frames,audio=audio,reference_sha256=sha(a.reference),candidate_sha256=sha(a.candidate),contract_sha256=sha(a.contract),seconds=time.monotonic()-start,human_review='not performed')
 if net is not None:r['metric_weights']={p.name:sha(p) for p in (Path(torch.hub.get_dir())/'checkpoints').glob('alexnet-*.pth')}|{'lpips-alex-v0.1':sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')}
 write(a.out/'result.json',r);print(json.dumps(checks));return 0 if r['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
