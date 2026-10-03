#!/usr/bin/env python3
# Historical evidence renderer only. This is not a current qualification gate.
"""Create a self-contained local gallery from retained CUDA campaign records.

No model, engine, latent or raw RGB files are required by the resulting pages.
Run after copying JSON/CSV/logs/MP4/PNG assets from the qualification node.
"""
import argparse, csv, datetime, hashlib, html, json, pathlib, re, statistics, subprocess
from html.parser import HTMLParser
from urllib.parse import unquote

def read(path, default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
def lines(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text(errors='replace').splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:pass
    return result
def esc(value):return html.escape(str(value),quote=True)
def number(value, digits=2):return '—' if value is None else f'{value:,.{digits}f}'
def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

CSS='''
:root{color-scheme:dark;font:16px/1.5 system-ui,sans-serif;background:#10151c;color:#e7edf5}
body{max-width:1500px;margin:auto;padding:24px}h1{font-size:2rem}h2{margin:0 0 12px}h3{font-size:1rem}
a{color:#9dcaff}p{max-width:1100px}section{background:#19212c;border:1px solid #354253;border-radius:12px;padding:20px;margin:24px 0}
.players,.stills{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}figure{margin:0}figcaption{padding:6px 0}
video,img{width:100%;background:#000;border-radius:6px}video{max-height:430px}.controls{display:flex;align-items:center;gap:12px;margin:14px 0}.controls input{flex:1}
button{padding:8px 14px;background:#31465f;color:white;border:1px solid #7891ad;border-radius:6px;cursor:pointer}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{padding:8px;text-align:left;border-bottom:1px solid #354253;vertical-align:top}.scroll{overflow:auto}
.pass{color:#83e1b5}.fail{color:#ffc177}.note{color:#b6c6d9;font-size:.9rem}.badge{background:#23374a;border-radius:4px;padding:3px 7px}summary{cursor:pointer;margin:12px 0}code{overflow-wrap:anywhere}
'''
JS='''
document.querySelectorAll('.comparison').forEach(section=>{
 const videos=[...section.querySelectorAll('video')], slider=section.querySelector('input'),button=section.querySelector('button'),label=section.querySelector('output');
 if(!videos.length)return;const lead=videos[0];let seeking=false;
 const seek=t=>{seeking=true;videos.forEach(v=>{if(Number.isFinite(v.duration))v.currentTime=Math.min(t,v.duration)});setTimeout(()=>seeking=false,80)};
 const pause=()=>videos.forEach(v=>{if(!v.paused)v.pause()});
 const play=()=>{document.querySelectorAll('video').forEach(v=>{if(!videos.includes(v))v.pause()});videos.forEach(v=>v.play().catch(()=>{}))};
 button.addEventListener('click',()=>lead.paused?play():pause());
 slider.addEventListener('input',()=>seek(Number(slider.value)));
 videos.forEach(v=>{v.addEventListener('play',play);v.addEventListener('pause',pause);v.addEventListener('seeking',()=>{if(!seeking)seek(v.currentTime)})});
 function tick(){if(Number.isFinite(lead.duration)){slider.max=lead.duration;slider.value=lead.currentTime;label.textContent=lead.currentTime.toFixed(2)+' / '+lead.duration.toFixed(2)+' s'}button.textContent=lead.paused?'Play together':'Pause together';if(!lead.paused&&!seeking)videos.slice(1).forEach(v=>{if(Math.abs(v.currentTime-lead.currentTime)>.12)v.currentTime=lead.currentTime});requestAnimationFrame(tick)}tick();
});
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/fast-vae/cuda');p.add_argument('--verify-media',action='store_true');a=p.parse_args()
    root=pathlib.Path(a.root);root.mkdir(parents=True,exist_ok=True);used=set();summary={}
    records={x.parent.name:read(x,{}) for x in root.glob('*/record.json')};ledger=read(root/'ledger.json',{})
    def link(path,label):return f'<a href="{esc(path)}">{esc(label)}</a>' if (root/path).is_file() else esc(label)+' (unavailable)'
    def player(path,label):
        used.add(path);return f'<figure><video controls playsinline muted preload="metadata" src="{esc(path)}"></video><figcaption>{esc(label)} · {link(path,"MP4")}</figcaption></figure>'
    def compare(items):return '<div class="controls"><button type="button">Play together</button><input aria-label="Shared video time" type="range" min="0" max="1" step="0.0416667" value="0"><output>0 s</output></div><div class="players">'+''.join(player(x,y) for x,y in items)+'</div>'
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3 faster VAE · CUDA review</title><style>'+CSS+'</style><body><h1>H3 original-model VAE · CUDA review</h1>',
      '<p>RTX PRO 6000 Blackwell Server Edition · SM120 · CUDA 12.8. Original checkpoint and unchanged denoiser. New generated clips complete <strong>six steps</strong>. Main gates: <strong>243 frames at 24 fps</strong>. Reference-derived fixtures and the 362-frame repeated-tail stress case run no additional denoising.</p>',
      '<p>Native balanced uses BF16 matrix operands with FP32 accumulation, softmax, normalization and residuals. Optional TensorRT uses FP16 matrices and FP32 sensitive operations. TAEH3 remains unchanged. These results do not qualify Metal (M7).</p>',
      '<p class="note">Metrics compare float RGB before codec loss. Playback is H.264 and is for inspection; quality-comparison MP4s are silent. Complete generation outputs below include audio. Players start muted; unmute one at a time. Controls synchronize each comparison group.</p>']
    if ledger:
        end=ledger.get('finished_unix') or max((x.get('started_unix',0)+x.get('wall_seconds',0) for x in ledger.get('entries',[])),default=ledger.get('started_unix',0))
        elapsed=(end-ledger.get('started_unix',end))/60
        parts.append(f'<p>Campaign elapsed: {elapsed:.1f} / 480 minutes (includes setup, failed runs and idle work). '+link('ledger.json','Campaign ledger')+' · '+link('gates.json','Frozen gates')+' · '+link('inventory.json','Hardware / source inventory')+' · '+link('audit.json','Artifact and invariance audit')+'</p>')
    performance=[]
    for case in ['C1','C2']:
        for name,job,mode in [('FP32',case+'-decode','reference'),('Native BF16',case+'-decode','balanced'),('TensorRT FP16',case+'-trt-decode','tensorrt'),('Legacy fast CUDA',case+'-legacy-fast','legacy'),('TAEH3 preview',case+'-tiny-bench','tiny')]:
            rows=[x for x in lines(root/job/'stdout.log') if x.get('mode')==mode]
            if not rows:continue
            warm=[x['decode_seconds'] for x in rows if x['repeat']>0];first=next(x for x in rows if x['repeat']==0)
            peak=first.get('peak_tensor_bytes')
            performance.append([case,name,number(first['load_seconds']),number(first['decode_seconds']),number(statistics.median(warm)) if warm else '—',f'{number(min(warm))}–{number(max(warm))}' if warm else '—',number(peak/2**30) if peak else 'sampled only',link(job+'/record.json','record')])
            summary.setdefault(case,{})[mode]=dict(load_seconds=first['load_seconds'],cold_decode_seconds=first['decode_seconds'],warm_seconds=warm,median_warm=statistics.median(warm) if warm else None,tracked_peak_bytes=peak)
    parts.append('<section><h2>Complete decoder performance</h2>'+table(['Case','Mode','Load (s)','First decode (s)','Warm median (s)','Warm range (s)','Tracked peak (GiB)','Evidence'],performance)+'<p class="note">Decode includes tile execution, readback, unpacking and host spatial/temporal stitching. First decode writes the raw comparison file; warm repeats discard the sink payload. FP32/BF16 pairs are interleaved with both decoders resident. Tracked peaks are per decoder; whole-device sampling during a pair includes both. TensorRT adds native and external allocator peaks conservatively. These timings exclude audio and mux; complete wall times follow.</p></section>')
    wall=[]
    for case in ['C1','C2']:
        for mode in ['reference','balanced','tensorrt','legacy']:
            id=case+'-final-'+mode;r=records.get(id,{})
            if not r:continue
            phase=dict(re.findall(r'h3(?:cli)?: phase duration ([^:\n]+): ([\d.]+) s',(root/id/'stderr.log').read_text(errors='replace')))
            wall.append([case,mode,esc(r.get('status')),f'{r.get("completed_steps",0)}/6',number(r.get('wall_seconds')),number(r.get('peak_vram_bytes',0)/2**30) if r.get('peak_vram_bytes') else 'unsampled',number(r.get('peak_rss_bytes',0)/2**30) if r.get('peak_rss_bytes') else 'unsampled',link(id+'/record.json','record')+' · '+link(id+'/stderr.log','phases')])
            summary.setdefault(case,{}).setdefault(mode,{})['end_to_end']=dict(wall_seconds=r.get('wall_seconds'),phase_seconds=phase,peak_vram_bytes=r.get('peak_vram_bytes'),peak_rss_bytes=r.get('peak_rss_bytes'),completed_steps=r.get('completed_steps'))
    parts.append('<section><h2>Six-step end-to-end wall time</h2>'+table(['Case','Mode','Status','Steps','Wall (s)','VRAM peak (GiB)','Host RSS peak (GiB)','Evidence'],wall)+'<p class="note">Complete process wall time includes startup, conditioning, denoising, audio, decoder loading and FFmpeg. VRAM is whole-device usage sampled at 1 Hz, not an exact allocator peak; test records and audit retain sample gaps. RSS sampling covers the main process, not all descendants. High-water RSS and peaks from short runs may be missed. No reference and candidate denoising are run concurrently.</p></section>')
    gains=[]
    for case in ['C1','C2']:
        data=summary.get(case,{});ref=data.get('reference',{});legacy=data.get('legacy',{})
        for mode in ['balanced','tensorrt']:
            candidate=data.get(mode,{})
            if not ref.get('median_warm') or not candidate.get('median_warm'):continue
            speed=ref['median_warm']/candidate['median_warm'];old=legacy.get('median_warm')
            base_wall=ref.get('end_to_end',{}).get('wall_seconds');wall_time=candidate.get('end_to_end',{}).get('wall_seconds');old_wall=legacy.get('end_to_end',{}).get('wall_seconds')
            gain=1-wall_time/base_wall if wall_time and base_wall else None
            candidate['gates']=dict(decode_speedup=speed,decode_pass=speed>=1.5 if case=='C1' else speed>=1/1.03,wall_gain=gain,wall_pass=gain>=.05 if gain is not None else None)
            gains.append([case,mode,number(speed)+'×',number(old/candidate['median_warm'])+'×' if old else '—',number(gain*100)+'%' if gain is not None else 'pending',number((1-wall_time/old_wall)*100)+'%' if old_wall and wall_time else 'pending'])
    parts.append('<section><h2>Measured gains</h2>'+table(['Case','Candidate','Decode vs FP32','Decode vs legacy fast','Wall saved vs FP32','Wall saved vs legacy fast'],gains)+'<p class="note">Cold means a new process/decoder with filesystem caches left intact. Engine hashing/loading remains included. End-to-end wall comparisons are single complete runs per final mode; warm decoder variation is shown separately.</p></section>')
    stress=[]
    for job in ['C3-repeat','C3-trt-decode']:
        record=records.get(job,{})
        for row in lines(root/job/'stdout.log'):
            if 'decode_seconds' not in row:continue
            stress.append([esc(row['mode']),str(row['repeat']),number(row['decode_seconds']),number(row['peak_tensor_bytes']/2**30),number(row.get('peak_rss_native',0)/2**20),number(record.get('peak_vram_bytes',0)/2**30),link(job+'/record.json','record')])
    if stress:parts.append('<section><h2>362-frame repeated-decode memory</h2>'+table(['Mode','Repeat','Decode (s)','Tracked peak (GiB)','Process high-water RSS (GiB)','Sampled device peak (GiB)','Evidence'],stress)+'<p class="note">Repeated-tail C3 state; decode only. Linux getrusage RSS is a process high-water mark, sampled device peak covers the whole job. The lifetime tests separately assert live allocation plateaus, cancellation and clean retry. Materialized float output is intentionally a separate API with full-clip host allocation.</p></section>')
    cases=[('C0','C0-quality','Small canvas / short temporal tail'),('C1','C1-quality','640×480 · 243 frames'),('C2','C2-quality','1344×768 · 243 frames'),('C3','C3-quality','1344×768 · 362-frame memory stress'),('C4 faces','C4-faces-balanced-quality','Reference-derived face, text and moving crop'),('C4 texture','C4-texture-balanced-quality','Reference-derived fine texture, text and moving crop')]
    quality=[]
    for case,directory,title in cases:
        data=read(root/directory/'metrics.json')
        if not data:
            parts.append('<section><h2>'+esc(case+' · '+title)+'</h2><p class="fail">Quality report unavailable; this case is not qualified.</p></section>');continue
        alternatives={'C1':'C1-trt-quality','C2':'C2-trt-quality','C3':'C3-trt-quality','C4 faces':'C4-faces-tensorrt-quality','C4 texture':'C4-texture-tensorrt-quality'}
        trtdir=alternatives.get(case);trt=read(root/trtdir/'metrics.json') if trtdir else None
        items=[(directory+'/reference.mp4','FP32 reference'),(directory+'/balanced.mp4','Native BF16 balanced'),(directory+'/tiny.mp4','TAEH3 preview, unchanged')]
        if trt:items.append((trtdir+'/balanced.mp4','Optional TensorRT FP16'))
        passed=all(data['gates'].values());quality.append(dict(case=case,native_gates=data['gates'],tensorrt_gates=trt['gates'] if trt else None))
        provenance='Verified six-step latent decode; no extra denoising.'
        if case=='C0':provenance='Spatial/time crop of the verified six-step C1 state; no extra denoising.'
        if case=='C3':provenance='Repeated-tail expansion of C2’s verified six-step state from 243 to 362 frames. Decode-only geometry/memory stress, not an independently generated 362-frame clip.'
        if case.startswith('C4'):provenance='Original encoder applied to a deterministic photo pan with text; denoiser-free. Source is retained in pixels/.'
        parts.append('<section class="comparison" id="'+esc(case.replace(' ','-'))+'"><h2>'+esc(case+' · '+title)+'</h2><p>'+esc(provenance)+' <span class="'+('pass' if passed else 'fail')+'">Native gates '+('PASS' if passed else 'FAIL')+'</span></p>'+compare(items))
        rows=[]
        for label,m in [('Native BF16',data['modes']['balanced']),('TAEH3',data['modes']['tiny'])]+([('TensorRT FP16',trt['modes']['balanced'])] if trt else []):
            rows.append([label,number(m['psnr']),number(m['ssim'],6),number(m['lpips'],7),number(m['worst_psnr']),str(m['worst_frame']),number(m.get('seam_worst_psnr'))])
        parts.append(table(['Mode vs FP32','PSNR dB','SSIM','LPIPS ↓','Worst frame dB','Worst index','Worst seam dB'],rows))
        parts.append('<p>'+link(directory+'/metrics.json','All metrics and input checksums')+' · '+link(directory+'/frames.csv','Every-frame / temporal-join CSV')+' · '+link(directory+'/record.json','Quality command')+' · '+link(directory+'/stderr.log','Log')+'</p>')
        worst=data['modes']['balanced']['worst_frame'];parts.append(f'<details><summary>Worst frame {worst} and difference ×20</summary><div class="stills">'+''.join('<figure><img loading="lazy" src="'+esc(directory+'/'+name)+'"><figcaption>'+esc(label)+'</figcaption></figure>' for name,label in [('reference-worst.png','FP32'),('balanced-worst.png','Native BF16'),('tiny-worst.png','TAEH3'),('difference-x20.png','Native minus reference, absolute ×20')])+'</div></details></section>')
        if trt:
            parts.append('<section><details><summary>'+esc(case)+' · TensorRT worst frame '+str(trt['modes']['balanced']['worst_frame'])+'</summary><p>'+link(trtdir+'/metrics.json','TensorRT metrics / gates')+' · '+link(trtdir+'/frames.csv','Every-frame CSV')+'</p><div class="stills">'+''.join('<figure><img loading="lazy" src="'+esc(trtdir+'/'+name)+'"><figcaption>'+esc(label)+'</figcaption></figure>' for name,label in [('reference-worst.png','FP32'),('balanced-worst.png','TensorRT FP16'),('difference-x20.png','Difference ×20')])+'</div></details></section>')
        if data.get('seam_crop'):
            crop=data['seam_crop'];parts.append('<section><h3>'+esc(case)+' · worst seam crop, frame '+str(crop['frame'])+'</h3><p class="note">'+esc(crop['selection'])+'; bounds '+esc(crop['bounds_xyxy'])+'</p><div class="stills">'+''.join('<figure><img loading="lazy" style="image-rendering:pixelated;max-width:256px" src="'+esc(directory+'/'+name)+'"><figcaption>'+esc(label)+'</figcaption></figure>' for name,label in [('seam-reference.png','FP32'),('seam-balanced.png','Native BF16'),('seam-tiny.png','TAEH3'),('seam-difference-x20.png','Difference ×20')])+'</div></section>')
        rgb=[]
        for label,m in [('Native BF16',data['modes']['balanced']),('TAEH3',data['modes']['tiny'])]+([('TensorRT FP16',trt['modes']['balanced'])] if trt else []):
            if 'max_abs' in m:rgb.append([label,number(m['max_abs'],7),number(m['changed_rgb8_fraction']*100)+'%',str(m['max_rgb8_difference']),number(m.get('max_temporal_error_rms'),7)])
        if rgb:parts.append('<section><h3>'+esc(case)+' · maximum and temporal errors</h3>'+table(['Mode','Max float error','Changed RGB8 channels','Max RGB8 code delta','Max frame-to-frame error RMS'],rgb)+'</section>')
    parts.append('<section><h2>Quality and performance gates</h2><p>Frozen thresholds: raw tile relative L2 ≤0.02; RGB PSNR ≥35 dB; SSIM ≥0.98; worst frame ≥30 dB; at least 3 dB over TAEH3 and LPIPS ≤80% of TAEH3. Every frame is measured at its native resolution. Seam measurements cover eight-pixel bands around tile starts; temporal joins are flagged in CSV. Targets: ≥1.5× C1 decode speedup, ≥5% complete wall-time gain, and no meaningful C2 decode regression.</p><p>FP32 is the local oracle. The independent PyTorch FP32 tile comparison and TensorRT FP32 export are retained in the evidence table. These results qualify the measured corpus, not arbitrary future inputs or different GPUs.</p></section>')
    audit=read(root/'audit.json',{})
    if audit:
        categories=[('Latent identity',audit.get('latent_invariance',[])),('Fusion / delivery identity',audit.get('delivery_and_fusion_bitwise',[])),('Media frames',audit.get('media',[])),('Six-step CLI provenance',audit.get('generated_steps',[])),('Six-step API provenance',audit.get('cached_api_steps',[])),('Required compatibility tests',audit.get('qualification_records',[]))]
        parts.append('<section><h2>Retained qualification audit</h2>'+table(['Check','Passed','Total'],[[esc(label),str(sum(bool(x.get('passed')) for x in values)),str(len(values))] for label,values in categories])+'<p>'+link('audit.json','Full checksums, sample gaps and per-case results')+'</p></section>')
        variations=audit.get('uncontrolled_ref2va_repeats',[])
        if variations:parts.append('<section><h2>Ref2VA reproducibility diagnostics</h2>'+table(['Image size','Comparison to original reference','Latent relative L2','Max latent error','Bitwise'],[[esc(x['case']),esc(x['comparison']),number(x['rel_l2'],7),number(x['max_abs'],7),str(x['bitwise'])] for x in variations])+'<p>These original runs used timing-based fast-CUDA GEMM selection. They are retained diagnostics, not decoder quality comparisons. The qualification audit separately requires paired encoder/decoder policy runs with <code>H3_FAST_CUDA_GEMM_TUNE=0</code>, plus exact posterior/RNG comparisons. All complete renders still use six steps.</p></section>')
    enc=[]
    for row in audit.get('encoder',[]):
        baseline=row.get('baseline',{});reuse=row.get('reuse',{})
        enc.append([esc(row['case']),number(baseline.get('seconds')),number(reuse.get('seconds')),str(baseline.get('allocations','—'))+' → '+str(reuse.get('allocations','—')),esc(str(row.get('moments_bitwise'))),esc(str(row.get('sample_bitwise'))),esc(str(row.get('epsilon_bitwise')))])
    if enc:parts.append('<section><h2>Reference encoder</h2>'+table(['Input','Baseline (s)','Reuse (s)','Allocations','Moments bitwise','Sample bitwise','RNG bitwise'],enc)+'<p>'+link('audit.json','Posterior, conditioning-state and memory audit')+'</p></section>')
    # Include every other completed MP4, including final audio/video outputs,
    # cached-session checks and denoiser-free source fixtures.
    remaining=sorted(str(x.relative_to(root)) for x in root.rglob('*.mp4') if str(x.relative_to(root)) not in used)
    parts.append('<h2>Generation, compatibility and source videos</h2><p>Generated clips below use six completed steps. The pixels/ and C5-video-inputs/ source clips are reference-derived and denoiser-free; decode-only continuation clips reuse existing six-step states. Initial versus final builds are distinguished by directory name and binary checksum in each record.</p>')
    groups={}
    for path in remaining:
        group=path.split('/')[0]+(' · shape change' if pathlib.Path(path).stem.startswith('shape-') else '')
        groups.setdefault(group,[]).append(path)
    for group,paths in groups.items():
        folder=paths[0].split('/')[0];r=records.get(folder,{});status=r.get('status','source/derived asset');steps=r.get('completed_steps')
        if folder in ('final-session','session-invariance','session-final-repeat') and status=='pass':steps=6
        parts.append('<section class="comparison"><h3>'+esc(group)+' · '+esc(status)+(f' · {steps}/6 steps' if steps is not None else '')+'</h3>'+compare([(x,pathlib.Path(x).stem) for x in paths])+'<p>'+link(folder+'/record.json','Command / timing / memory')+' · '+link(folder+'/stdout.log','Output')+' · '+link(folder+'/stderr.log','Log')+'</p></section>')
    auditrows=[]
    for name,r in sorted(records.items(),key=lambda item:item[1].get('started_unix',0)):
        auditrows.append([esc(name),'<span class="'+('pass' if r.get('status')=='pass' else 'fail')+'">'+esc(r.get('status','unknown'))+'</span>',number(r.get('wall_seconds')),link(name+'/record.json','command / result')+' · '+link(name+'/stdout.log','stdout')+' · '+link(name+'/stderr.log','stderr')])
    parts.append('<section><h2>Experiments and complete evidence ledger</h2><p>CUDA Graph capture was rejected by cuBLASLt on this node; the opt-in path disables capture and retries ordinary execution. Batch 4 lost throughput; batch 2’s small gain did not justify doubled decoder residency. Batch 1 remains selected. Fusion saves 42 MiB of scratch with roughly equal tile latency; pinned staging also showed no independent tile-speed win. Same-stream asynchronous copies preserve ordering; no cross-tile overlap speedup is claimed.</p><p>Early allocator-alignment and single-tile reporting failures are retained with their successful retests. Host continuation/bridge tests initially hit the generation-only evaluation cap; host-only retests remove that cap. The resume harness initially supplied fast_cuda explicitly, which the existing checkpoint API rejects; its retest lets the checkpoint restore execution. The initial multiple-video fixture was shorter than the released two-second minimum; its retest uses two 48-frame reference clips. Audits run before these retests remain recorded as incomplete. The default stays legacy; TensorRT remains optional and requires a GPU/runtime/checkpoint-matched engine.</p>'+table(['Test / experiment','Status','Wall (s)','Evidence'],auditrows)+'</section>')
    parts.append('<script>'+JS+'</script></body></html>');(root/'review.html').write_text('\n'.join(parts))
    summary['quality']=quality;(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    index='<!doctype html><html lang="en"><meta charset="utf-8"><title>H3 VAE reports</title><style>'+CSS+'</style><body><h1>H3 faster original-model VAE</h1>'+''.join('<p><a href="'+backend+'/review.html">'+label+'</a></p>' for backend,label in [('cuda','CUDA playback and qualification'),('metal','Metal M4 playback and qualification')] if (root.parent/backend/'review.html').is_file())+'<p>Independent backend measurements; quality, memory and speed gates are reported separately. TAEH3 preview is unchanged.</p></body></html>'
    (root.parent/'index.html').write_text(index)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('src','href') and value and not value.startswith(('#','http:','https:')):self.paths.append(unquote(value))
    validations=[]
    for page in [root/'review.html',root.parent/'index.html']:
        parser=Links();parser.feed(page.read_text());missing=[x for x in parser.paths if not (page.parent/x).is_file()];assert not missing,(page,missing)
        validations.append(dict(page=str(page),checked_local_links=len(parser.paths),missing=missing))
    media=[]
    if a.verify_media:
        for path in sorted(root.rglob('*.mp4')):
            cmd=['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_type,width,height,nb_read_frames,duration,r_frame_rate','-of','json',str(path)]
            probe=json.loads(subprocess.check_output(cmd,text=True));video=next(x for x in probe['streams'] if x['codec_type']=='video');assert int(video['nb_read_frames'])>0
            media.append(dict(path=str(path.relative_to(root)),streams=probe['streams']))
    (root/'report-validation.json').write_text(json.dumps(dict(generator_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),pages=validations,media=media,synchronized_controls='shared play/pause, native seeking, shared slider, 120 ms drift correction; offline inline JavaScript'),indent=2)+'\n')
    print(root/'review.html');print(len(records),'records;',len(used),'videos;',sum(x['checked_local_links'] for x in validations),'local links checked')
if __name__=='__main__':main()
