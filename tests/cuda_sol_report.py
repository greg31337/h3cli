#!/usr/bin/env python3
"""GPU-free evidence report; every row links to its immutable request and log."""
import csv,html,json,subprocess,time,re
import numpy as np
from pathlib import Path
from cuda_sol_campaign import ROOT,records,clock,remaining
from cuda_multi_reference import sha,save

def timed_steps(row):
    log=(ROOT/'runs'/row['id']/'stderr.log').read_text(errors='replace')
    return {int(step):float(seconds) for step,seconds in re.findall(r'denoise\s+(\d+)\s*/\s*\d+[^\r\n]*?([0-9]+\.[0-9]+)\s*s',log)}

def order(ident):return (0 if ident.startswith('R') else 1 if ident.startswith('E') else 2 if ident.startswith('F') else 3,ident)

def main():
    rows=records();render=json.loads((ROOT/'report-renders.json').read_text())
    fields=['id','status','mode','minimum','width','height','frames','delivered_frames','wall_seconds','startup_approx_seconds','denoise_seconds','post_denoise_approx_seconds','step1_seconds','step2_seconds','peak_device_gib','peak_process_gib','peak_host_gib','max_sample_gap','exact_pairs','approximate_pairs','approximate_fraction','planned_workspace_bytes','workspace_bytes','gpu_released']
    flat=[]
    for r in render:
        c=r['case'];t=r.get('telemetry',{});peaks=t.get('peaks_bytes',{});counts=(r.get('sol_counters') or [{}])[-1];steps=r.get('step_seconds',[])
        numbered=timed_steps(r)
        startup=next((e['seconds'] for e in r.get('phase_events',[]) if 'starting' in e['line']),None)
        flat.append(dict(id=r['id'],status=r['status'],mode=r['mode'],minimum=r['minimum'],width=c['width'],height=c['height'],frames=c['frames'],delivered_frames=c['frames']-(39 if c.get('continue_from') else 0),wall_seconds=r['wall_seconds'],startup_approx_seconds=startup,denoise_seconds=sum(steps),post_denoise_approx_seconds=r['wall_seconds']-sum(steps)-startup if steps and startup is not None else None,step1_seconds=numbered.get(1),step2_seconds=numbered.get(2),peak_device_gib=peaks.get('device',0)/2**30,peak_process_gib=peaks.get('process',0)/2**30 if t.get('process_counter_available') else None,peak_host_gib=peaks.get('rss',0)/2**30,max_sample_gap=t.get('max_gap_seconds'),exact_pairs=counts.get('exact'),approximate_pairs=counts.get('approximate'),approximate_fraction=counts.get('approximate',0)/max(1,counts.get('approximate',0)+counts.get('exact',0)) if counts else None,planned_workspace_bytes=counts.get('workspace'),workspace_bytes=counts.get('reserved'),gpu_released=r.get('gpu_released')))
    with (ROOT/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
    save(ROOT/'results.json',flat)
    pairs=[]
    for ident in sorted({r['case']['id'] for r in render},key=order):
        p={r['mode']:r for r in render if r['case']['id']==ident}
        if not all(k in p for k in ['default','sol']):continue
        a,b=p['default'],p['sol'];valid=all(r['status']=='pass' and r.get('media',{}).get('valid') for r in (a,b));pair=dict(id=ident,valid=valid,dense=a['wall_seconds'],sol=b['wall_seconds'],width=a['case']['width'],frames=a['case']['frames'])
        if valid and not ident.startswith('F-'):pair['wall_speedup']=a['wall_seconds']/b['wall_seconds']
        if valid and not ident.startswith('F-') and len(a['step_seconds'])>1 and len(b['step_seconds'])>1:pair['routed_step_speedup']=a['step_seconds'][-1]/b['step_seconds'][-1]
        d=ROOT/'runs'/a['id']/'output.mp4';s=ROOT/'runs'/b['id']/'output.mp4'
        if d.exists() and s.exists() and remaining(True)>180:
            log=ROOT/(ident+'-ssim.log')
            if not log.exists():
                with log.open('w') as f:subprocess.run(['ffmpeg','-hide_banner','-i',str(d),'-i',str(s),'-lavfi','ssim','-f','null','-'],stdout=f,stderr=f,timeout=120,check=True)
            pair['ssim_log']=log.name
            match=re.search(r'All:([0-9.]+)',log.read_text())
            pair['ssim']=float(match[1]) if match else None
            audio=[]
            for path in (d,s):
                pcm=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-vn','-f','f32le','-acodec','pcm_f32le','-'],capture_output=True,check=True,timeout=60).stdout
                audio.append(np.frombuffer(pcm,dtype='<f4').astype('float64'))
            n=min(map(len,audio));x,y=(a[:n] for a in audio)
            pair['audio']=dict(decoded_sample_counts=list(map(len,audio)),rms_dense=float(np.sqrt(np.mean(x*x))),rms_sol=float(np.sqrt(np.mean(y*y))),difference_rms=float(np.sqrt(np.mean((x-y)**2))),relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-30)),cosine=float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-30)))
        if ident.startswith('F-'):pair['performance_scope']='Composition only: dense includes first cache packing, SOL reuses the cache'
        else:pair['performance_scope']='Single matched two-step sample; provisional'
        pairs.append(pair)
    save(ROOT/'pairs.json',pairs)
    save(ROOT/'report-metadata.json',{'generated_epoch':time.time(),'lpips':'Not collected: SSIM and audio diagnostics are used; no perceptual-quality qualification from two-step outputs','report_sha256':sha(__file__),'phase_precision':'Startup is sampled at approximately 1-second intervals. Post-denoise combines decoding and delivery; it is not isolated VAE time.'})
    pre='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>CUDA SOL review</title><style>body{font:16px system-ui;background:#15171b;color:#eee;margin:30px}a{color:#8bc7ff}section{margin:30px 0;padding:20px;background:#22252b}video{width:100%;max-height:500px}.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:20px}pre{white-space:pre-wrap}td,th{padding:7px;text-align:left}img{max-width:180px;max-height:140px}</style><h1>CUDA SOL review</h1><p>Two-step diagnostic videos; these do not establish final rendering quality. Dense and SOL use the same prompt, seed and references. Failed and missing gates remain visible.</p><p><a href="index.html">Index</a> · <a href="640.html">640×480</a> · <a href="1344.html">1344×768</a> · <a href="results.csv">CSV</a> · <a href="coverage.json">Coverage</a></p>'''
    for width in [None,640,1344]:
        body=[pre]
        coverage=json.loads((ROOT/'coverage.json').read_text())
        body.append('<p>Automated closing gates: '+('PASS' if coverage['passed'] else 'INCOMPLETE / FAILED')+'. Final quality remains unqualified. Runtime defaults remain unchanged; measured SOL candidate uses min_exact=0.5.</p>')
        body.append('<table><tr><th>Case</th><th>Dense wall</th><th>SOL wall</th><th>Wall speedup</th><th>Routed-step speedup</th></tr>')
        for pair in pairs:
            if width and pair['width']!=width:continue
            speed=f"{pair['wall_speedup']:.3f}×" if 'wall_speedup' in pair else '—';step=f"{pair['routed_step_speedup']:.3f}×" if 'routed_step_speedup' in pair else '—'
            body.append(f'<tr><td>{pair["id"]}</td><td>{pair["dense"]:.2f}s</td><td>{pair["sol"]:.2f}s</td><td>{speed}</td><td>{step}</td></tr>')
        body.append('</table><p>NVFP4 smoke timings include unmatched cold cache preparation and carry no comparative speed claim. <a href="pairs.json">SSIM/audio diagnostics</a> · <a href="heldout-velocity.json">Teacher-forced velocity</a></p>')
        for ident in sorted({r['case']['id'] for r in render if width is None or r['case']['width']==width},key=order):
            group=[r for r in render if r['case']['id']==ident];c=group[0]['case'];body.append(f'<section><h2>{html.escape(ident)} — {c["width"]}×{c["height"]}, {c["frames"]} frames</h2><p>{html.escape(c["prompt"])}</p><div class="pair">')
            for r in group:
                prefix='runs/'+r['id'];path=ROOT/prefix/'output.mp4'
                body.append(f'<article><h3>{html.escape(r["mode"])} · {r["status"]} · {r["wall_seconds"]:.2f}s</h3>')
                if r.get('superseded_by'):body.append(f'<p>Original failed attempt retained; superseded by {html.escape(r["superseded_by"])} after recorded cold fingerprint preparation. <a href="resume-retest.json">Retry record</a></p>')
                if path.exists():body.append(f'<video controls preload="metadata" src="{prefix}/output.mp4"></video>')
                data=next(x for x in flat if x['id']==r['id'])
                steps=', '.join(f'{i}: {x:.2f}s' for i,x in sorted(timed_steps(r).items())) or 'decode only / not recorded'
                body.append(f'<p>Steps: {steps}. Peak device VRAM: {data["peak_device_gib"]:.3f} GiB. Peak host RSS: {data["peak_host_gib"]:.3f} GiB.</p><p><a href="{prefix}/record.json">Metrics and command</a> · <a href="{prefix}/stderr.log">Log</a> · <a href="{prefix}/final-media.json">Media checks</a></p>')
                sheet=ROOT/prefix/'contact.jpg'
                if path.exists() and not sheet.exists() and remaining(True)>120:
                    delivered=c['frames']-(39 if c.get('continue_from') else 0)
                    subprocess.run(['ffmpeg','-v','error','-y','-i',str(path),'-vf',f'fps=288/{delivered},scale=240:-1,tile=4x3','-frames:v','1',str(sheet)],capture_output=True,timeout=60,check=True)
                if sheet.exists():body.append(f'<a href="{prefix}/contact.jpg"><img src="{prefix}/contact.jpg" alt="Contact sheet"></a>')
                body.append('</article>')
            body.append('</div><button onclick="this.closest(\'section\').querySelectorAll(\'video\').forEach((v,i)=>{v.currentTime=0;v.muted=i>0;v.play()})">Play together</button>')
            if (ROOT/'identity.json').exists():
                assets=json.loads((ROOT/'identity.json').read_text())['assets']
                for key in c.get('references',[]):
                    a=assets[key];ref=Path(a['path']).relative_to(ROOT)
                    link=html.escape(str(ref));body.append(f'<a href="{link}">{key}</a> ')
                    if a['kind']=='image':body.append(f'<img src="{link}" alt="{key}">')
            body.append('</section>')
        name='index.html' if width is None else str(width)+'.html';(ROOT/name).write_text(''.join(body))
    exclusions={'.h3av','.h3sample','.bf16','.qkv','.f32','.bin'}
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix not in exclusions and p.name not in ['checksums.json','runner.lock'] and 'diagnostics/' not in str(p)}
    save(ROOT/'checksums.json',files)
    print(f'Reported {len(render)} renders and {len(pairs)} pairs')
if __name__=='__main__':main()
