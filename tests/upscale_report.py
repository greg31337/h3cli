#!/usr/bin/env python3
"""CPU diagnostics and synchronized playback of the existing 14 campaign videos."""
import argparse
import csv
import hashlib
import html
from html.parser import HTMLParser
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from cuda_reference_regression import sha, write
from upscale_campaign import IDS, manifest, media


def frames(path, size, env):
    w,h=size
    p=subprocess.Popen([env.get('H3_FFMPEG','ffmpeg'),'-v','error','-i',str(path),'-an','-pix_fmt','rgb24','-f','rawvideo','-'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for _ in range(90):
            b=p.stdout.read(w*h*3);assert len(b)==w*h*3
            yield np.frombuffer(b,np.uint8).reshape(h,w,3)
        assert not p.stdout.read(1) and not p.stderr.read() and p.wait()==0
    finally:
        if p.poll() is None:p.terminate();p.wait()


def resize(x,size):
    return np.asarray(Image.fromarray(x).resize(tuple(size),Image.Resampling.BOX))


def gray(x):
    return (x.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32))/255.


def detail(x):
    g=gray(x)
    lap=g[:-2,1:-1]+g[2:,1:-1]+g[1:-1,:-2]+g[1:-1,2:]-4*g[1:-1,1:-1]
    return float(np.mean(lap.astype(np.float64)**2))


def waveform(path,env):
    b=subprocess.check_output([env.get('H3_FFMPEG','ffmpeg'),'-v','error','-i',str(path),'-vn','-acodec','pcm_f32le','-f','f32le','-'],env=env)
    a=np.frombuffer(b,'<f4');assert len(a)%2==0 and np.isfinite(a).all()
    return dict(sha256=hashlib.sha256(b).hexdigest(),samples_per_channel=len(a)//2,rms=float(np.sqrt(np.mean(a.astype(np.float64)**2))))


def single_frame(path,index,size,env):
    b=subprocess.check_output([env.get('H3_FFMPEG','ffmpeg'),'-v','error','-i',str(path),'-vf',f'select=eq(n\\,{index})','-frames:v','1','-an','-pix_fmt','rgb24','-f','rawvideo','-'],env=env)
    w,h=size;assert len(b)==w*h*3;return Image.fromarray(np.frombuffer(b,np.uint8).reshape(h,w,3))


def metrics(path,size,low,direct,low_size,target_size,d0,out,env):
    rows=[];previous=None;worst=[]
    for i,x in enumerate(frames(path,size,env)):
        y=resize(x,low_size);delta=(y.astype(np.float32)-low[i].astype(np.float32))/255.
        temporal=0. if previous is None else float(np.sqrt(np.mean((delta-previous).astype(np.float64)**2)))
        previous=delta
        r=dict(frame=i,seconds=i/24,source_rmse=float(np.sqrt(np.mean(delta.astype(np.float64)**2))),
               source_ssim=float(structural_similarity(gray(low[i]),gray(y),data_range=1.)),
               direct_ssim_diagnostic=float(structural_similarity(gray(direct[i]),gray(y),data_range=1.)),
               temporal_delta_rmse=temporal,native_laplacian_energy=detail(x))
        rows.append(r)
        if len(worst)<3 or temporal>min(z[0] for z in worst):
            worst.append((temporal,i,Image.fromarray(x).copy()));worst=sorted(worst,key=lambda z:(-z[0],z[1]))[:3]
    out.mkdir(parents=True,exist_ok=True);images=[]
    tw,th=target_size;cw,ch=min(512,tw),min(320,th);box=((tw-cw)//2,(th-ch)//2,(tw+cw)//2,(th+ch)//2)
    for _,i,candidate in worst:
        source=Image.fromarray(low[i]).resize((tw,th),Image.Resampling.LANCZOS)
        target=candidate.resize((tw,th),Image.Resampling.LANCZOS)
        baseline=single_frame(d0,i,target_size,env)
        panel=Image.new('RGB',(cw*3,ch))
        for k,img in enumerate((source,target,baseline)):panel.paste(img.crop(box),(k*cw,0))
        name=f'frame-{i:02d}.png';panel.save(out/name);images.append(name)
    result=dict(frames=rows,crops=images,summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('frame','seconds')})
    result['summary']['temporal_delta_max']=max(r['temporal_delta_rmse'] for r in rows)
    write(out/'metrics.json',result);return result


def publish(out,report):
    rows=report['cases'];fields=['pair','id','later_job_seconds','end_to_end_seconds','speed_ratio_to_direct','peak_vram_gib','source_ssim','temporal_delta_rmse','detail_ratio_to_pixel','pcm_exact','aac_exact']
    with (out/'report.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
    table=''.join('<tr onclick="choose('+str(i)+')">'+''.join('<td>'+html.escape(f'{r[k]:.4g}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in fields)+'</tr>' for i,r in enumerate(rows))
    options=''.join(f'<option value="{i}">{r["pair"]} / {r["id"]}</option>' for i,r in enumerate(rows))
    links=''.join(f'<li>{r["pair"]}/{r["id"]}: <a href="{r["video"]}">video</a> · <a href="{r["directory"]}/result.json">command/timing</a> · <a href="{r["log"]}">raw log</a> · <a href="{r["metrics"]}">frame diagnostics</a></li>' for r in rows)
    text='''<!doctype html><html><head><meta charset="utf-8"><title>h3cli latent upscaling</title><style>
body{font:16px system-ui;margin:24px;background:#101820;color:#edf3f8}a{color:#8bcbff}video{width:48%;background:black}select,button{font:inherit;margin:8px;padding:6px}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:8px;border:1px solid #3d5364;text-align:right}tr:hover{background:#273d4c;cursor:pointer}img{max-width:100%}.muted{color:#c0ccd5}code{overflow-wrap:anywhere}</style></head><body>
<h1>Native latent upscaling</h1><p>672×384 → 1344×768 and 960×544 → 1920×1088 · 90 frames · 24 FPS · seed 42 · 50 source/direct steps · dense BF16.</p>
<p>14 existing videos, one per variant and canvas. Human visual review: <b>pending</b>. Refinement uses sigma 0.25; I4/U2/U4 share the persisted noise. Audio latents and decoded PCM are preserved exactly.</p>
<p class="muted">The direct high-resolution render can depict a different scene despite sharing the seed. Similarity scores are diagnostics, not ground-truth fidelity or visual approval. Detail energy can reward noise and ringing. No confidence interval or general speed/quality claim follows from one prompt and seed.</p>
<label>Candidate <select id="selection" onchange="choose(+this.value)">OPTIONS</select></label><label>Compare with <select id="reference" onchange="choose(+document.getElementById('selection').value)"><option>D0</option><option>P0</option><option>L0</option></select></label>
<button onclick="playBoth()">Play together</button><button onclick="pauseBoth()">Pause</button><label>Audio <select onchange="audio(this.value)"><option>candidate</option><option>reference</option><option>muted</option></select></label><p id="label"></p>
<video id="referenceVideo" controls preload="metadata" muted></video> <video id="candidateVideo" controls preload="metadata"></video>
<h2>Measured workflow costs and diagnostics</h2><p>Later-job costs include initialization, render and each job's full share of measured transfer/load work. End-to-end adds source generation with preview decoding removed; P0 includes the source decode it needs. Jobs ran as separate native processes with normal OS caching, not guaranteed cold filesystem caches. Verification PCM decodes and report computation are excluded. Speed ratio is direct wall / end-to-end cost; greater than one is faster.</p>
<table><thead><tr>HEADERS</tr></thead><tbody>TABLE</tbody></table>
<p><a href="report.json">Complete JSON</a> · <a href="report.csv">CSV</a> · <a href="identity.json">Frozen identities/environment</a> · <a href="manifest.json">Protocol</a> · <a href="ledger.json">Stage accounting</a> · <a href="attempts.jsonl">All attempts</a></p>
<h2>Largest temporal source discrepancies</h2><p>Native center crops, left to right: Lanczos source, selected candidate, direct high-resolution render. Selection uses the largest difference in consecutive-frame changes after area downsampling to source size. It is not an automatic ranking of visual defects.</p><div id="crops"></div><h2>Artifacts</h2><ul>LINKS</ul>
<script>const rows=DATA;const b=document.getElementById('referenceVideo'),c=document.getElementById('candidateVideo');
function pauseBoth(){b.pause();c.pause()}function playBoth(){b.currentTime=c.currentTime;Promise.all([b.play(),c.play()]).catch(console.error)}function audio(v){b.muted=v!=='reference';c.muted=v!=='candidate'}
function choose(i){pauseBoth();let r=rows[i],ref=document.getElementById('reference').value;document.getElementById('selection').value=i;b.src=r.pair+'/'+ref+'/video.mp4';c.src=r.video;document.getElementById('label').textContent=r.pair+': '+ref+' (left), '+r.id+' (right)';document.getElementById('crops').innerHTML=r.crops.map(p=>'<p><img alt="Source, candidate and direct render center crops" src="'+p+'"></p>').join('')}
c.addEventListener('seeking',()=>{b.currentTime=c.currentTime});c.addEventListener('ratechange',()=>{b.playbackRate=c.playbackRate});c.addEventListener('pause',()=>b.pause());c.addEventListener('play',()=>{b.currentTime=c.currentTime;b.play().catch(console.error)});setInterval(()=>{if(!c.paused&&Math.abs(b.currentTime-c.currentTime)>.1)b.currentTime=c.currentTime},100);choose(0);audio('candidate');</script></body></html>'''
    text=text.replace('OPTIONS',options).replace('HEADERS',''.join('<th>'+k+'</th>' for k in fields)).replace('TABLE',table).replace('LINKS',links).replace('DATA',json.dumps(rows).replace('</','<\\/'))
    (out/'index.html').write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);a=p.parse_args();out=a.out.resolve();env=os.environ.copy()
    ledger=json.loads((out/'ledger.json').read_text());assert ledger['complete'] and len(ledger['cases'])==14
    protocol=manifest(out/'manifest.json');identity=json.loads((out/'identity.json').read_text())
    assert sha(out/'manifest.json')==identity['manifest_sha256']
    rows=[]
    for pair in protocol['pairs']:
        name=pair['id'];base=out/name;low=list(frames(base/'L0/video.mp4',pair['source'],env))
        direct=[resize(x,pair['source']) for x in frames(base/'D0/video.mp4',pair['target'],env)]
        source_pcm=sha(base/'verify-L0-pcm/audio.f32');source_aac=waveform(base/'L0/video.mp4',env)
        pair_rows=[]
        for case in IDS:
            d=base/case;record=json.loads((d/'result.json').read_text());assert record['passed']
            for file,digest in record['artifacts'].items():assert sha(d/file)==digest
            entry=next(c for c in ledger['cases'] if c['pair']==name and c['id']==case);assert sha(d/'video.mp4')==entry['sha256']
            size=pair['source'] if case=='L0' else pair['target'];media(d/'video.mp4',size,env)
            md=out/'metrics'/name/case
            metric=metrics(d/'video.mp4',size,low,direct,pair['source'],pair['target'],base/'D0/video.mp4',md,env)
            aac=waveform(d/'video.mp4',env);aac_exact=aac['sha256']==source_aac['sha256']
            pcm_exact=None if case=='P0' else sha(base/f'verify-{case}-pcm/audio.f32')==source_pcm
            if case not in ('L0','D0','P0'):assert pcm_exact
            if case=='P0':assert aac_exact
            costs=ledger['costs'][name][case];direct_cost=ledger['costs'][name]['D0']['end_to_end_seconds']
            r=dict(pair=name,id=case,directory=str(d.relative_to(out)),video=entry['output'],video_sha256=entry['sha256'],
                   log=record['attempt']+'/run.log',metrics=str((md/'metrics.json').relative_to(out)),crops=[str((md/f).relative_to(out)) for f in metric['crops']],
                   **costs,speed_ratio_to_direct=direct_cost/costs['end_to_end_seconds'],peak_vram_gib=record['peak_vram_bytes']/1024**3,
                   pcm_exact=pcm_exact,aac_exact=aac_exact,aac=aac,**metric['summary'])
            pair_rows.append(r);print(name,case,r['source_ssim'],flush=True)
        pixel=next(r for r in pair_rows if r['id']=='P0')['native_laplacian_energy']
        for r in pair_rows:r['detail_ratio_to_pixel']=None if r['id']=='L0' else r['native_laplacian_energy']/max(pixel,1e-30)
        rows+=pair_rows
    report=dict(schema=1,human_visual_review='pending',identity=identity,protocol=protocol,cases=rows,
                metrics_identity=dict(source_sha256=sha(Path(__file__)),packages={p:importlib.metadata.version(p) for p in ('numpy','Pillow','scikit-image')}),
                limitations=['One prompt/seed and one observation per variant.','Direct cross-resolution renders are not paired ground truth.',
                    'Source consistency and temporal delta can penalize legitimate detail changes; detail energy can reward noise.',
                    'Timings describe separate-process stages with shared work fully charged, not guaranteed cold caches.',
                    'No official Regenerate-2K parity, quantization, approximate sampling or general perceptual qualification.'])
    write(out/'report.json',report);publish(out,report)
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('src','href') and value and not value.startswith(('http:','https:','#')):assert (out/value).is_file(),value
    Links().feed((out/'index.html').read_text())
    assert len(rows)==14 and all((out/c).is_file() for r in rows for c in r['crops'])
    write(out/'report-audit.json',dict(passed=True,videos=14,crops=sum(len(r['crops']) for r in rows),
        index_sha256=sha(out/'index.html'),report_sha256=sha(out/'report.json'),csv_sha256=sha(out/'report.csv'),human_visual_review='pending'))


if __name__=='__main__':main()
