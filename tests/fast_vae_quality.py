#!/usr/bin/env python3
"""Every-frame float-RGB metrics and full playback assets, before codec loss."""
import argparse,csv,json,pathlib,subprocess,time,hashlib,math
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch,lpips

def psnr(delta):
    mse=float(np.mean(np.square(delta,dtype=np.float64)))
    return -10*np.log10(mse) if mse>0 else 999.
def video(path,raw,w,h):
    command=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','24','-i','pipe:0','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    p=subprocess.Popen(command,stdin=subprocess.PIPE)
    for frame in raw:p.stdin.write(np.rint(np.clip(frame,0,1)*255).astype(np.uint8).tobytes())
    p.stdin.close()
    if p.wait():raise RuntimeError('FFmpeg failed')
    return command

def main():
    p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('balanced');p.add_argument('tiny');p.add_argument('output');p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True);p.add_argument('--frames',type=int,required=True);p.add_argument('--cpu',action='store_true');p.add_argument('--device',choices=['cpu','cuda','mps'],default='cuda');a=p.parse_args()
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True);shape=(a.frames,a.height,a.width,3)
    files=dict(reference=a.reference,balanced=a.balanced,tiny=a.tiny);values={};inputs={}
    for name,path in files.items():
        path=pathlib.Path(path);assert path.stat().st_size==np.prod(shape)*4,(path,path.stat().st_size,shape)
        values[name]=np.memmap(path,dtype='<f4',mode='r',shape=shape);assert np.isfinite(values[name]).all(),name
        inputs[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
    device='cpu' if a.cpu else a.device;torch.set_num_threads(4);net=lpips.LPIPS(net='alex',version='0.1').to(device).eval()
    rows=[];previous={};start=time.monotonic()
    # Native 256px planner: distribute extra overlap in round-robin 16px units.
    seam=np.zeros((a.height,a.width),bool)
    for axis,size in [(0,a.height),(1,a.width)]:
        count=max(1,math.ceil((size-256)/192)+1);overlap=[64]*(count-1)
        if count==1:continue
        for j in range(max(0,(256*count-64*(count-1)-size)//16)):overlap[j%(count-1)]+=16
        starts=[0]
        for x in overlap:starts.append(starts[-1]+256-x)
        for pos in starts[1:]:
            if axis==0:seam[max(0,pos-4):pos+4,:]=True
            else:seam[:,max(0,pos-4):pos+4]=True
    for i in range(a.frames):
        ref=np.asarray(values['reference'][i]);tr=torch.from_numpy(ref.copy()).permute(2,0,1).unsqueeze(0).to(device)*2-1
        for name in ['balanced','tiny']:
            x=np.asarray(values[name][i]);d=x-ref
            codes=np.rint(x*255).astype(np.int16)-np.rint(ref*255).astype(np.int16)
            temporal=float(np.sqrt(np.mean(np.square(d-previous[name],dtype=np.float64)))) if name in previous else None
            previous[name]=d
            with torch.inference_mode():perceptual=float(net(tr,torch.from_numpy(x.copy()).permute(2,0,1).unsqueeze(0).to(device)*2-1).item())
            rows.append(dict(frame=i,mode=name,mse=float(np.mean(np.square(d,dtype=np.float64))),psnr=float(psnr(d)),ssim=float(structural_similarity(ref,x,data_range=1,channel_axis=-1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)),lpips=perceptual,seam_psnr=float(psnr(d[seam])) if seam.any() else None,temporal_join=(i>=17 and (i-17)%17<5),max_abs=float(abs(d).max()),changed_rgb8_fraction=float(np.mean(codes!=0)),max_rgb8_difference=int(abs(codes).max()),temporal_error_rms=temporal))
        if i%24==0:print(f'metrics {i+1}/{a.frames}',flush=True)
    with (out/'frames.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    result=dict(width=a.width,height=a.height,frames=a.frames,fps=24,inputs=inputs,metric_definition='F32 clamped/blended native RGB before encoding; PSNR data_range=1; Gaussian SSIM sigma=1.5 population covariance; LPIPS AlexNet v0.1 full native resolution; every frame',metric_seconds=time.monotonic()-start,torch=torch.__version__,device=device,modes={})
    for name in ['balanced','tiny']:
        r=[x for x in rows if x['mode']==name];worst=min(r,key=lambda x:x['psnr'])
        result['modes'][name]={k:float(np.mean([x[k] for x in r])) for k in ['psnr','ssim','lpips']}
        result['modes'][name].update(worst_frame=worst['frame'],worst_psnr=worst['psnr'],seam_worst_psnr=min((x['seam_psnr'] for x in r if x['seam_psnr'] is not None),default=None))
        result['modes'][name]['mean_frame_psnr']=result['modes'][name]['psnr']
        result['modes'][name].update(max_abs=max(x['max_abs'] for x in r),changed_rgb8_fraction=float(np.mean([x['changed_rgb8_fraction'] for x in r])),max_rgb8_difference=max(x['max_rgb8_difference'] for x in r),max_temporal_error_rms=max((x['temporal_error_rms'] for x in r if x['temporal_error_rms'] is not None),default=None))
        mse=float(np.mean([x['mse'] for x in r]));result['modes'][name]['psnr']=float(-10*np.log10(mse)) if mse else 999.
    result['metric_weights']={str(x.name):hashlib.file_digest(x.open('rb'),'sha256').hexdigest() for x in (pathlib.Path(torch.hub.get_dir())/'checkpoints').glob('alexnet-*.pth')}
    calibrated=pathlib.Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
    result['metric_weights']['lpips_v0.1_alex.pth']=hashlib.file_digest(calibrated.open('rb'),'sha256').hexdigest()
    b=result['modes']['balanced'];tiny=result['modes']['tiny']
    result['gates']={'psnr':b['psnr']>=35,'ssim':b['ssim']>=.98,'worst_frame':b['worst_psnr']>=30,'better_than_tiny_psnr':b['psnr']>=tiny['psnr']+3,'better_than_tiny_lpips':b['lpips']<=.8*tiny['lpips']}
    worst=b['worst_frame'];result['video_commands']={}
    for name,raw in values.items():
        Image.fromarray(np.rint(np.clip(raw[worst],0,1)*255).astype(np.uint8)).save(out/(name+'-worst.png'))
        result['video_commands'][name]=video(out/(name+'.mp4'),raw,a.width,a.height)
    Image.fromarray(np.rint(np.clip(abs(values['balanced'][worst]-values['reference'][worst])*20,0,1)*255).astype(np.uint8)).save(out/'difference-x20.png')
    if seam.any():
        seam_frame=min((x for x in rows if x['mode']=='balanced'),key=lambda x:x['seam_psnr'])['frame']
        delta=abs(values['balanced'][seam_frame]-values['reference'][seam_frame]);score=delta.max(-1)*seam
        y,x=np.unravel_index(score.argmax(),score.shape);y0=max(0,int(y)-32);x0=max(0,int(x)-32);y1=min(a.height,y0+64);x1=min(a.width,x0+64)
        for name,raw in values.items():Image.fromarray(np.rint(raw[seam_frame,y0:y1,x0:x1]*255).astype(np.uint8)).save(out/('seam-'+name+'.png'))
        Image.fromarray(np.rint(np.clip(delta[y0:y1,x0:x1]*20,0,1)*255).astype(np.uint8)).save(out/'seam-difference-x20.png')
        result['seam_crop']=dict(frame=seam_frame,bounds_xyxy=[x0,y0,x1,y1],selection='largest error within tile-start bands in worst seam-PSNR frame')
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result['gates']))
if __name__=='__main__':main()
