#!/usr/bin/env python3
"""Reference-only controls and paired perceptual screens; no renderer changes.

Use the pinned requirements-metal-quality environment. LPIPS runs on CPU.
Thresholds describe this operational screen, not human perceptual equivalence.
"""
import argparse
import hashlib
import html
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess

import cv2
import lpips
import numpy as np
import torch
from scipy.signal import stft

ROOT=Path(__file__).resolve().parents[1]
cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def decode(path):
    probe=json.loads(subprocess.run(['ffprobe','-v','error','-show_streams','-of','json',str(path)],
                                   capture_output=True,text=True,check=True).stdout)
    video=next(s for s in probe['streams'] if s['codec_type']=='video')
    audio=next(s for s in probe['streams'] if s['codec_type']=='audio')
    rate=video['r_frame_rate'].split('/');fps=float(rate[0])/float(rate[1])
    if (video['width'],video['height'])!=(640,480) or abs(fps-24)>1e-6:
        raise ValueError('Expected 640x480 at 24 fps')
    raw=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-vf','scale=320:240:flags=area',
                        '-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True).stdout
    rgb=np.frombuffer(raw,np.uint8).reshape(-1,240,320,3).copy()
    if len(rgb)!=243:raise ValueError('Expected exactly 243 decoded frames')
    raw=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-map','0:a:0','-f','f32le','-ar','32000','-ac','2','pipe:1'],
                       capture_output=True,check=True).stdout
    pcm=np.frombuffer(raw,'<f4').reshape(-1,2).copy()
    if not np.isfinite(pcm).all():raise ValueError('Nonfinite decoded audio')
    return rgb,pcm,{'fps':fps,'frames':len(rgb),'video_seconds':float(video['duration']),
                    'audio_seconds':float(audio['duration']),'video_start':float(video.get('start_time',0)),
                    'audio_start':float(audio.get('start_time',0))}

class Metrics:
    def __init__(self,contract):
        for name,wanted in contract['metric_packages'].items():
            if version(name)!=wanted:raise ValueError('Metric package mismatch: '+name)
        checkpoint=Path(os.environ.get('TORCH_HOME',str(Path.home()/'.cache/torch')))/'hub/checkpoints/squeezenet1_1-b8a52dc0.pth'
        head=Path(lpips.__file__).parent/'weights/v0.1/squeeze.pth'
        for path,key in ((checkpoint,checkpoint.name),(head,'lpips-v0.1-squeeze.pth')):
            if sha(path)!=contract['metric_weights_sha256'][key]:raise ValueError('Metric weight checksum mismatch: '+key)
        self.net=lpips.LPIPS(net='squeeze',version='0.1',verbose=False).eval().cpu()
        model_root=checkpoint.parents[2]/'opencv'
        for name,digest in contract['face_models_sha256'].items():
            if sha(model_root/name)!=digest:raise ValueError('Face model checksum mismatch: '+name)
        self.face=cv2.FaceDetectorYN_create(str(model_root/'face_detection_yunet_2026may.onnx'),'',(320,240),.8,.3,5000)
        self.identity=cv2.FaceRecognizerSF_create(str(model_root/'face_recognition_sface_2021dec.onnx'),'')
        self.contract=contract
    def faces(self,rgb):
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        self.face.setInputSize((rgb.shape[1],rgb.shape[0]))
        _,rows=self.face.detect(bgr)
        return bgr,[] if rows is None else rows
    def distance(self,a,b):
        x=torch.from_numpy(np.stack(a).copy()).permute(0,3,1,2).float()/127.5-1
        y=torch.from_numpy(np.stack(b).copy()).permute(0,3,1,2).float()/127.5-1
        with torch.inference_mode():return self.net(x,y).flatten().numpy().tolist()
    def video(self,a,b,indices,face_required):
        a=a[indices];b=b[indices];dist=[];ssim=[];flow=[];mouth=[];face_dist=[];identities=[];detected=matched=0
        for start in range(0,len(a),8):dist.extend(self.distance(a[start:start+8],b[start:start+8]))
        ga=np.stack([cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in a]);gb=np.stack([cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in b])
        for i,(x,y) in enumerate(zip(ga,gb)):
            xf=x.astype(np.float32)/255;yf=y.astype(np.float32)/255
            ux=cv2.GaussianBlur(xf,(11,11),1.5);uy=cv2.GaussianBlur(yf,(11,11),1.5)
            vx=np.maximum(0,cv2.GaussianBlur(xf*xf,(11,11),1.5)-ux*ux)
            vy=np.maximum(0,cv2.GaussianBlur(yf*yf,(11,11),1.5)-uy*uy)
            cov=cv2.GaussianBlur(xf*yf,(11,11),1.5)-ux*uy
            ssim.append(float(np.mean(((2*ux*uy+.01**2)*(2*cov+.03**2))/
                                     ((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2)))))
            if i:
                f=cv2.calcOpticalFlowFarneback(ga[i-1],x,None,.5,3,15,3,5,1.2,0)
                g=cv2.calcOpticalFlowFarneback(gb[i-1],y,None,.5,3,15,3,5,1.2,0)
                flow.append(float(np.mean(np.linalg.norm(f-g,axis=2))))
            if face_required:
                bgr_a,faces=self.faces(a[i]);bgr_b,others=self.faces(b[i])
                if len(faces):
                    detected+=1;box=max(faces,key=lambda z:z[2]*z[3]);xx,yy,w,h=map(int,box[:4])
                    xx=max(0,xx);yy=max(0,yy);w=min(w,x.shape[1]-xx);h=min(h,x.shape[0]-yy)
                    def iou(z):
                        x1,y1,w1,h1=map(int,z[:4]);inter=max(0,min(xx+w,x1+w1)-max(xx,x1))*max(0,min(yy+h,y1+h1)-max(yy,y1))
                        return inter/max(1,w*h+w1*h1-inter)
                    if w>0 and h>0 and len(others) and max(map(iou,others))>=.5:
                        matched+=1
                        other=max(others,key=iou)
                        fa=self.identity.feature(self.identity.alignCrop(bgr_a,box)).copy()
                        fb=self.identity.feature(self.identity.alignCrop(bgr_b,other)).copy()
                        identities.append(float(self.identity.match(fa,fb,cv2.FaceRecognizerSF_FR_COSINE)))
                        face_dist.extend(self.distance([cv2.resize(a[i,yy:yy+h,xx:xx+w],(96,96))],
                                                       [cv2.resize(b[i,yy:yy+h,xx:xx+w],(96,96))]))
                        if i:
                            mouth.append((int(indices[i]),float(np.mean(np.abs(ga[i,yy+h//2:yy+h,xx:xx+w].astype(float)-ga[i-1,yy+h//2:yy+h,xx:xx+w]))),
                                          float(np.mean(np.abs(gb[i,yy+h//2:yy+h,xx:xx+w].astype(float)-gb[i-1,yy+h//2:yy+h,xx:xx+w])))))
        delta=(np.diff(ga.astype(np.float32),axis=0)-np.diff(gb.astype(np.float32),axis=0))/255
        return {'lpips_mean':float(np.mean(dist)),'lpips_p95':float(np.quantile(dist,.95)),
                'ssim_mean':float(np.mean(ssim)),'ssim_p05':float(np.quantile(ssim,.05)),
                'temporal_delta_rmse':float(np.sqrt(np.mean(delta*delta))) if len(delta) else 0,
                'flow_endpoint_mean_px':float(np.mean(flow)) if flow else 0,
                'face_reference_detections':detected,'face_matched_fraction':matched/max(1,detected),
                'face_lpips_p95':float(np.quantile(face_dist,.95)) if face_dist else None,
                'face_identity_cosine_p05':float(np.quantile(identities,.05)) if identities else None,
                'sample_indices':list(map(int,indices)),'mouth_motion':mouth}

def correlation(a,b):
    if np.std(a)<1e-9 or np.std(b)<1e-9:return float(np.allclose(a,b,atol=1e-7))
    return float(np.corrcoef(a,b)[0,1])

def audio_metrics(a,b):
    difference=len(a)-len(b);n=min(len(a),len(b));a=a[:n].astype(np.float64);b=b[:n].astype(np.float64)
    if n<1024 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Missing/nonfinite audio')
    env=[]
    for x in (a,b):env.append(np.array([np.sqrt(np.mean(x[round(i*32000/24):round((i+1)*32000/24)]**2)) for i in range(int(n*24/32000))]))
    spectra=[]
    for x in (a,b):
        # Preserve channels so opposite stereo signals cannot cancel.
        spectra.append(np.stack([np.log10(np.maximum(np.abs(stft(x[:,ch],fs=32000,nperseg=1024,noverlap=768,boundary=None)[2]),1e-4)) for ch in (0,1)]))
    rel=float(np.linalg.norm(env[1]-env[0])/max(np.linalg.norm(env[0]),1e-9))
    return {'log_spectrum_mean_abs':float(np.mean(np.abs(spectra[0]-spectra[1]))),
            'envelope_relative_l2':rel,'envelope_correlation':correlation(*env),
            'clipped_fraction_increase':float(max(0,np.mean(np.abs(b)>=.999)-np.mean(np.abs(a)>=.999))),
            'new_dropout_windows':int(np.sum((env[0]>.002)&(env[1]<.0001))),
            'reference_envelope':env[0].tolist(),'candidate_envelope':env[1].tolist(),
            'sample_count_difference':difference}

def assess(metrics,limits,face_required=False):
    v=metrics['video'];a=metrics['audio'];checks={}
    for key in ('lpips_mean','lpips_p95','temporal_delta_rmse'):
        checks[key]=v[key]<=limits[key+'_max']
    checks['flow']=v['flow_endpoint_mean_px']<=limits['flow_endpoint_mean_max_px']
    for key in ('ssim_mean','ssim_p05'):checks[key]=v[key]>=limits[key+'_min']
    if face_required:
        checks['face_coverage']=v['face_reference_detections']>=limits['face_reference_detections_min'] and v['face_matched_fraction']>=limits['face_matched_fraction_min']
        checks['face_appearance']=v['face_lpips_p95'] is not None and v['face_lpips_p95']<=limits['face_lpips_p95_max']
        checks['face_identity']=v['face_identity_cosine_p05'] is not None and v['face_identity_cosine_p05']>=limits['face_identity_cosine_p05_min']
        checks['av_proxy']=metrics.get('av_proxy_lag_difference') is not None and metrics['av_proxy_lag_difference']<=limits['av_proxy_lag_difference_max_seconds']
    for key in ('log_spectrum_mean_abs','envelope_relative_l2','clipped_fraction_increase','new_dropout_windows'):
        checks['audio_'+key]=a[key]<=limits['audio_'+key+'_max']
    checks['audio_envelope_correlation']=a['envelope_correlation']>=limits['audio_envelope_correlation_min']
    checks['audio_sample_count']=a['sample_count_difference']==0
    checks['av_timing']=metrics['av_timing_difference']<=limits['av_duration_difference_max_seconds']
    if 'boundary' in metrics:
        checks['boundary']=all(metrics['boundary'][k]<=limits[k+'_max'] for k in ('lpips_mean','lpips_p95')) and metrics['boundary']['ssim_mean']>=limits['ssim_mean_min'] and metrics['boundary']['ssim_p05']>=limits['ssim_p05_min']
    return {'checks':checks,'pass':all(checks.values())}

def measure(engine,ra,rb,aa,ab,meta_a,meta_b,face=False,prefix=0):
    indices=sorted(set(range(prefix,243,4))|{242})
    result={'video':engine.video(ra,rb,indices,face),'audio':audio_metrics(aa[round(prefix*32000/24):],ab[round(prefix*32000/24):]),
            'av_timing_difference':max(abs(meta_a[k]-meta_b[k]) for k in ('video_seconds','audio_seconds','video_start','audio_start'))}
    if prefix:
        indices=list(range(max(0,prefix-6),min(243,prefix+13)))
        result['boundary']=engine.video(ra,rb,indices,False)
    if face:
        mouth=result['video']['mouth_motion'];lags=[]
        for candidate in (False,True):
            env=np.array(result['audio']['candidate_envelope' if candidate else 'reference_envelope'])
            scores=[]
            for lag in range(-6,7):
                pairs=[(row[2 if candidate else 1],env[row[0]-prefix+lag]) for row in mouth if 0<=row[0]-prefix+lag<len(env)]
                scores.append(correlation(*np.array(pairs).T) if len(pairs)>=3 else -2.)
            lags.append(int(np.argmax(scores))-6 if max(scores)>-2 else None)
        result['av_proxy_lag_difference']=abs(lags[0]-lags[1])/24 if None not in lags else None
        result['av_proxy_lags_frames']=lags
    result.update(assess(result,engine.contract['limits'],face));return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path);p.add_argument('--candidate',type=Path)
    p.add_argument('--contract',type=Path,default=ROOT/'tests/metal_reference_contract.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--calibrate',action='store_true')
    p.add_argument('--face',action='store_true');p.add_argument('--prefix',type=int,choices=[0,39,90,141,192],default=0)
    a=p.parse_args()
    if a.output.exists():p.error('use a fresh output directory')
    a.output.mkdir(parents=True)
    contract=json.loads(a.contract.read_text());engine=Metrics(contract)
    ra,aa,ma=decode(a.reference)
    report={'contract_sha256':sha(a.contract),'metric_script_sha256':sha(__file__),'reference_sha256':sha(a.reference),
            'reference_metadata':ma,'results':{},'perceptual_observation':'automated screens only; no human observation claimed'}
    if a.calibrate:
        if a.candidate:p.error('calibration must not use candidate data')
        repeat,repeat_audio,repeat_meta=decode(a.reference)
        if not np.array_equal(ra,repeat) or not np.array_equal(aa,repeat_audio):raise ValueError('Reference decoding is not repeatable')
        rng=np.random.default_rng(7301)
        positive=np.clip(ra.astype(np.int16)+rng.integers(-1,2,ra.shape),0,255).astype(np.uint8)
        controls=[('repeat',repeat,repeat_audio,repeat_meta,True),('bounded-noise',positive,np.round(aa*32768)/32768,ma,True),
                  ('blur',np.stack([cv2.GaussianBlur(f,(0,0),3) for f in ra]),aa,ma,False),
                  ('translate',np.roll(ra,16,axis=2),aa,ma,False),('reverse',ra[::-1],aa,ma,False),
                  ('silence',ra,np.zeros_like(aa),ma,False),('audio-shift',ra,np.concatenate([np.zeros_like(aa[:8000]),aa[:-8000]]),ma,False)]
        for name,rb,ab,mb,expected in controls:
            r=measure(engine,ra,rb,aa,ab,ma,mb);r['expected_pass']=expected
            report['results'][name]=r
            (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        face_path=ROOT/contract['face_control_asset']
        if sha(face_path)!=contract['source_assets_sha256'][contract['face_control_asset']]:
            raise ValueError('Changed face calibration asset')
        face=cv2.resize(cv2.cvtColor(cv2.imread(str(face_path)),cv2.COLOR_BGR2RGB),(320,240))
        original=np.stack([face]*3)
        report['face_controls']={}
        for name,altered,expected in (
            ('repeat',original.copy(),True),
            ('bounded-noise',np.clip(original.astype(np.int16)+rng.integers(-1,2,original.shape),0,255).astype(np.uint8),True),
            ('missing-face',np.zeros_like(original),False)):
            v=engine.video(original,altered,[0,1,2],True);limits=contract['limits']
            passed=(v['face_reference_detections']>=limits['face_reference_detections_min'] and
                    v['face_matched_fraction']>=limits['face_matched_fraction_min'] and
                    v['face_lpips_p95'] is not None and v['face_lpips_p95']<=limits['face_lpips_p95_max'] and
                    v['face_identity_cosine_p05'] is not None and v['face_identity_cosine_p05']>=limits['face_identity_cosine_p05_min'])
            report['face_controls'][name]={'video':v,'pass':passed,'expected_pass':expected}
        report['calibration_pass']=all(v['pass']==v['expected_pass'] for v in report['results'].values())
        report['calibration_pass'] &= all(v['pass']==v['expected_pass'] for v in report['face_controls'].values())
    else:
        if not a.candidate:p.error('comparison requires --candidate')
        rb,ab,mb=decode(a.candidate)
        report['candidate_sha256']=sha(a.candidate);report['candidate_metadata']=mb
        report['results']['candidate']=measure(engine,ra,rb,aa,ab,ma,mb,a.face,a.prefix)
        report['screen_pass']=report['results']['candidate']['pass']
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    def relative(path):return os.path.relpath(path.resolve(),a.output.resolve())
    body=['<!doctype html><meta charset="utf-8"><title>M2 Reference quality screens</title>',
          '<style>body{font:16px system-ui;background:#171920;color:#eee;max-width:1300px;margin:2rem auto}.pair{display:flex;gap:1rem}video{width:48%}a{color:#9cd3ff}pre{white-space:pre-wrap}</style>',
          '<h1>M2 Reference quality screens</h1><p>243 frames, production VAE. Automated screening is distinct from actual human perceptual observations.</p>',
          '<div class="pair"><video controls src="'+html.escape(relative(a.reference))+'"></video>']
    if a.candidate:body+=['<video controls src="'+html.escape(relative(a.candidate))+'"></video>']
    body+=['</div><pre>'+html.escape(json.dumps(report,indent=2))+'</pre><p><a href="metrics.json">Metrics, controls and hashes</a></p>']
    (a.output/'review.html').write_text('\n'.join(body)+'\n')
    print(json.dumps({k:report[k] for k in ('calibration_pass','screen_pass') if k in report}))
    if not report.get('calibration_pass',report.get('screen_pass',False)):raise SystemExit(3)

if __name__=='__main__':main()
