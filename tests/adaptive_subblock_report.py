#!/usr/bin/env python3
"""Measure the twelve existing videos and publish an inspectable local report."""
import argparse
import csv
from collections import Counter
import html
from html.parser import HTMLParser
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
from scipy.signal import stft
from skimage.metrics import structural_similarity
from cuda_reference_regression import sha, write
from cuda_adaptive_subblock import IDS, media, av_state


def waveform(path,env):
    raw=subprocess.check_output([env.get('H3_FFMPEG','ffmpeg'),'-v','error','-i',str(path),'-vn','-acodec','pcm_f32le','-f','f32le','-'],env=env)
    x=np.frombuffer(raw,'<f4').reshape(-1,2);assert np.isfinite(x).all();return x


def audio_metrics(x,y):
    assert x.shape==y.shape
    xd,yd=x.astype(np.float64),y.astype(np.float64);diff=yd-xd
    norm=np.linalg.norm(xd);ynorm=np.linalg.norm(yd)
    _,_,sx=stft(xd,fs=32000,nperseg=1024,noverlap=768,axis=0)
    _,_,sy=stft(yd,fs=32000,nperseg=1024,noverlap=768,axis=0)
    mx,my=abs(sx),abs(sy)
    def levels(z):return dict(rms=np.sqrt(np.mean(z*z,axis=0)).tolist(),peak=np.max(abs(z),axis=0).tolist(),
                              clipping_fraction=np.mean(abs(z)>=.999,axis=0).tolist(),silence_fraction=np.mean(abs(z)<1e-4,axis=0).tolist())
    return dict(samples_per_channel=len(x),sample_rate=32000,channels=2,seconds=len(x)/32000,
                relative_l2=float(np.linalg.norm(diff)/max(norm,1e-30)),cosine=float(np.sum(xd*yd)/max(norm*ynorm,1e-30)),
                rms_error=float(np.sqrt(np.mean(diff*diff))),max_abs=float(abs(diff).max()),
                spectral_relative_l2=float(np.linalg.norm(my-mx)/max(np.linalg.norm(mx),1e-30)),
                log_spectral_rmse_db=float(np.sqrt(np.mean((20*np.log10(np.maximum(my,1e-5))-20*np.log10(np.maximum(mx,1e-5)))**2))),
                baseline=levels(xd),candidate=levels(yd))


def frames(reference,candidate,out,env,net,torch):
    out.mkdir(exist_ok=True);dec=[];result=[];previous=None;worst=[]
    try:
        for path in (reference,candidate):
            dec.append(subprocess.Popen([env.get('H3_FFMPEG','ffmpeg'),'-v','error','-i',str(path),'-an','-pix_fmt','rgb24','-f','rawvideo','-'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE))
        for i in range(90):
            raw=[d.stdout.read(640*480*3) for d in dec];assert all(len(r)==640*480*3 for r in raw)
            x,y=[np.frombuffer(r,np.uint8).reshape(480,640,3).astype(np.float32)/255 for r in raw];delta=y-x
            mse=float(np.mean(delta.astype(np.float64)**2));psnr=None if not mse else -10*math.log10(mse)
            if raw[0]==raw[1]:ssim=1.;lpips=0.
            else:
                ssim=float(structural_similarity(x,y,data_range=1,channel_axis=-1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False))
                with torch.inference_mode():lpips=float(net(*(torch.from_numpy(z).permute(2,0,1)[None]*2-1 for z in (x,y))).item())
            temporal=0. if previous is None else float(np.sqrt(np.mean((delta-previous).astype(np.float64)**2)))
            previous=delta
            result.append(dict(frame=i,seconds=i/24,ssim=ssim,psnr_db=psnr,lpips=lpips,temporal_error_rms=temporal))
            if len(worst)<3 or ssim<max(z[0] for z in worst):
                panel=np.concatenate((x,y,np.minimum(abs(delta)*10,1)),axis=1)
                worst.append((ssim,i,np.rint(panel*255).astype(np.uint8)));worst=sorted(worst,key=lambda z:(z[0],z[1]))[:3]
        for d in dec:assert not d.stdout.read(1) and not d.stderr.read() and d.wait()==0
    finally:
        for d in dec:
            if d.poll() is None:d.terminate();d.wait()
    images=[]
    for _,i,panel in worst:
        name=f'worst-{i:02d}.png';Image.fromarray(panel).save(out/name);images.append(name)
    return dict(frames=result,worst_images=images,summary=dict(ssim_min=min(x['ssim'] for x in result),ssim_mean=float(np.mean([x['ssim'] for x in result])),
                lpips_max=max(x['lpips'] for x in result),lpips_mean=float(np.mean([x['lpips'] for x in result])),
                psnr_mean_db=None if any(x['psnr_db'] is None for x in result) else float(np.mean([x['psnr_db'] for x in result])),
                temporal_max=max(x['temporal_error_rms'] for x in result)))


def publish(out,rows,identity,metrics_identity):
    report=dict(schema=1,workload=dict(width=640,height=480,frames=90,steps=50,seed=42),human_review='pending',default_changes='none',
                limitations=['One video per variant: single observations, no confidence intervals.','Only the recorded prompt, seed and SM120 device are qualified.',
                             'References, continuation, LoRA, quantization and Metal approximate execution remain unsupported.',
                             'Advisory page-cache preparation does not guarantee identical physical residency.',
                             'Metrics measure differences, not fidelity or perceptual acceptance.'],
                identity=identity,metrics_identity=metrics_identity,variants=rows)
    write(out/'report.json',report)
    fields=['id','wall_seconds','speed_ratio','denoise_seconds','attention_seconds','router_seconds','blocks','hits','sparse_calls','density','peak_vram_gib','peak_rss_gib','ssim_min','lpips_max','audio_relative_l2','historical_similarity_pass']
    with (out/'report.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
    table=''.join('<tr onclick="choose('+str(i)+')">'+''.join('<td>'+html.escape(f'{r[k]:.4g}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in fields)+'</tr>' for i,r in enumerate(rows))
    options=''.join(f'<option value="{i}">{r["id"]}</option>' for i,r in enumerate(rows))
    links=''.join(f'<li><a href="{r["directory"]}/video.mp4">{r["id"]} video</a> · <a href="{r["directory"]}/final.h3av">AV state</a> · <a href="{r["directory"]}/result.json">timing and command</a> · <a href="{r["directory"]}/steps.json">steps</a> · <a href="metrics/{r["id"]}/result.json">quality metrics</a></li>' for r in rows)
    text='''<!doctype html><html><head><meta charset="utf-8"><title>h3cli adaptive cache / SubBlock</title><style>
body{font:16px system-ui;margin:28px;background:#101820;color:#e9eff5}a{color:#82caff}video{width:48%;background:black}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:8px;border:1px solid #3e5262;text-align:right}tr:hover{background:#283b4b;cursor:pointer}button,select{font:inherit;margin:12px 5px;padding:6px}img{max-width:100%}.muted{color:#b9c9d6}</style></head><body>
<h1>Native adaptive cache and SubBlock</h1><p>640×480 · 90 frames · 50 steps · seed 42 · one video per variant. Both features remain off by default. Human review: <b>pending</b>.</p>
<p class="muted">Single observations on one SM120 device. Ratios are D0 wall time / candidate wall time. Similarity thresholds are historical diagnostics; failing them is reported, and passing them is not a parity or perceptual guarantee.</p>
<select id="selection" onchange="choose(+this.value)">OPTIONS</select><button onclick="playBoth()">Play both</button><button onclick="stopBoth()">Pause both</button><label>Audio <select onchange="audio(this.value)"><option>candidate</option><option>baseline</option><option>muted</option></select></label><p>D0 baseline (left) · <span id="label"></span> (right)</p>
<video id="baseline" controls preload="metadata" muted></video> <video id="candidate" controls preload="metadata"></video>
<h2>Measured results</h2><table><thead><tr>HEADERS</tr></thead><tbody>TABLE</tbody></table><p><a href="report.json">Complete JSON</a> · <a href="report.csv">CSV</a> · <a href="identity.json">Frozen source/build/runtime identity</a> · <a href="ledger.json">All attempts</a></p>
<h2>Worst frames</h2><p>Each image: D0, candidate, absolute difference ×10. Frames are selected by lowest SSIM.</p><div id="worst"></div><h2>Artifacts</h2><ul>LINKS</ul>
<script>const rows=DATA;const b=document.getElementById('baseline'),c=document.getElementById('candidate');b.src=rows[0].directory+'/video.mp4';
function stopBoth(){b.pause();c.pause()}function playBoth(){b.currentTime=c.currentTime;Promise.all([b.play(),c.play()]).catch(console.error)}
function audio(v){b.muted=v!=='baseline';c.muted=v!=='candidate'}function choose(i){stopBoth();document.getElementById('selection').value=i;let r=rows[i];c.src=r.directory+'/video.mp4';document.getElementById('label').textContent=r.id;document.getElementById('worst').innerHTML=r.worst_images.map(p=>'<img alt="Worst frame comparison" src="'+p+'">').join('')}
c.addEventListener('seeking',()=>{b.currentTime=c.currentTime});c.addEventListener('ratechange',()=>{b.playbackRate=c.playbackRate});c.addEventListener('pause',()=>b.pause());c.addEventListener('play',()=>{b.currentTime=c.currentTime;b.play().catch(console.error)});setInterval(()=>{if(!c.paused&&Math.abs(b.currentTime-c.currentTime)>.12)b.currentTime=c.currentTime},100);choose(0);audio('candidate');</script></body></html>'''
    text=text.replace('OPTIONS',options).replace('HEADERS',''.join('<th>'+k+'</th>' for k in fields)).replace('TABLE',table).replace('LINKS',links).replace('DATA',json.dumps(rows).replace('</','<\\/'))
    (out/'index.html').write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);a=p.parse_args();out=a.out.resolve();env=dict(os.environ)
    ledger=json.loads((out/'ledger.json').read_text());assert ledger['complete'] and [r['id'] for r in ledger['cases']]==IDS
    cases=ledger['cases'];identity=json.loads((out/'identity.json').read_text())
    import torch,lpips
    torch.set_num_threads(4);net=lpips.LPIPS(net='alex',version='0.1').cpu().eval()
    metric_identity=dict(packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-image','torch','torchvision','lpips','Pillow']},
        weights={p.name:sha(p) for p in (Path(torch.hub.get_dir())/'checkpoints').glob('alexnet-*.pth')}|{'lpips-alex-v0.1':sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')})
    assert len(metric_identity['weights'])>=2
    baseline=out/cases[0]['artifact_directory'];base_record=json.loads((baseline/'result.json').read_text());base_audio=waveform(baseline/'video.mp4',env)
    thresholds=json.loads((Path(__file__).parent/'cuda_reference/fast-quality.json').read_text())['gates'];rows=[]
    for case in cases:
        d=out/case['artifact_directory'];record=json.loads((d/'result.json').read_text());assert record['passed']
        for name,digest in record['artifacts'].items():assert sha(d/name)==digest
        probe=media(d/'video.mp4',env);av_state(d/'final.h3av')
        metrics=out/'metrics'/case['id'];metrics.mkdir(parents=True,exist_ok=True);path=metrics/'result.json'
        if path.exists():q=json.loads(path.read_text());assert q['video_sha256']==sha(d/'video.mp4') and q['metric_identity']==metric_identity
        else:
            start=time.monotonic();q=frames(baseline/'video.mp4',d/'video.mp4',metrics,env,net,torch)
            q.update(audio=audio_metrics(base_audio,waveform(d/'video.mp4',env)),video_sha256=sha(d/'video.mp4'),metric_identity=metric_identity,seconds=time.monotonic()-start)
            q['historical_similarity_checks']=dict(ssim=q['summary']['ssim_min']>=thresholds['every_frame_ssim_min'],lpips=q['summary']['lpips_max']<=thresholds['every_frame_lpips_alex_v01_max'],temporal=q['summary']['temporal_max']<=thresholds['temporal_error_rms_max'],audio=q['audio']['relative_l2']<=thresholds['audio_relative_l2_max'] and q['audio']['cosine']>=thresholds['audio_cosine_min'])
            q['human_review']='pending';write(path,q)
        steps=json.loads((d/'steps.json').read_text());cache=json.loads((d/'cache.json').read_text());counts=record['counts'];possible=counts['possible']
        row=dict(id=case['id'],directory=case['artifact_directory'],wall_seconds=record['wall_seconds'],speed_ratio=base_record['wall_seconds']/record['wall_seconds'],
            denoise_seconds=sum(s['wall_seconds'] for s in steps),attention_seconds=sum(s['attention_seconds'] for s in steps),router_seconds=sum(s['router_seconds'] for s in steps),
            blocks=counts['blocks'],hits=counts['hits'],sparse_calls=counts['sparse_calls'],density=counts['selected']/possible if possible else 1.,
            peak_vram_gib=record['peak_vram_bytes']/1024**3,peak_rss_gib=record['peak_host_rss_bytes']/1024**3,ssim_min=q['summary']['ssim_min'],lpips_max=q['summary']['lpips_max'],
            audio_relative_l2=q['audio']['relative_l2'],historical_similarity_pass=all(q['historical_similarity_checks'].values()),
            worst_images=[str((metrics/n).relative_to(out)) for n in q['worst_images']],stage_seconds=record['stage_seconds'],preparation=record['preparation'],
            peak_tensor_bytes=max(s['peak_tensor_bytes'] for s in steps),peak_pinned_bytes=max(s['peak_pinned_bytes'] for s in steps),
            h2d_bytes=sum(s['h2d_bytes'] for s in steps),d2h_bytes=sum(s['d2h_bytes'] for s in steps),router_calls=sum(s['router_calls'] for s in steps),
            dense_calls=counts['dense_calls'],protected_calls=sum(s['protected_calls'] for s in steps),
            dense_bypass=sum(s['dense_bypass'] for s in steps),cache_reasons=dict(Counter(c['reason'] for c in cache)),
            fallback_reasons={'warmup':sum(s['dense_bypass'] for s in steps[:10]),'dense_probe':sum(s['dense_bypass'] for s in steps[10:])},
            fallback_scope='Fixed eligible 640x480/90 workload: warmup before index 10, then dense block 0; protected query ranges counted separately',
            timing_missed=sum(s['attention_timing_missed'] for s in steps),sample_gap=record['maximum_sample_gap_seconds'],quality_summary=q['summary'],audio=q['audio'])
        rows.append(row);print(json.dumps({k:row[k] for k in ['id','speed_ratio','ssim_min','lpips_max','audio_relative_l2']}),flush=True)
    publish(out,rows,identity,metric_identity)
    # Every local URL must resolve; no media are regenerated or copied into alternate variants.
    class LocalLinks(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for name,link in attrs:
                if name in ('href','src') and link:assert (out/link).is_file(),link
    LocalLinks().feed((out/'index.html').read_text())
    for row in rows:
        assert (out/row['directory']/'video.mp4').is_file()
        assert all((out/name).is_file() for name in row['worst_images'])
    assets={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='assets.json'}
    write(out/'assets.json',assets)

if __name__=='__main__':main()
