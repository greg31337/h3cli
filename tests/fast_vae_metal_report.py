#!/usr/bin/env python3
# Historical evidence renderer only. This is not a current qualification gate.
"""Offline M4 gallery; performance failures and missing evidence stay visible."""
import argparse,csv,hashlib,json,pathlib,re,statistics,subprocess,time
from html.parser import HTMLParser
from urllib.parse import unquote
from fast_vae_report import CSS,JS,read,lines,esc,number,table

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='outputs/fast-vae/metal');parser.add_argument('--verify-media',action='store_true');a=parser.parse_args()
    root=pathlib.Path(a.root);root.mkdir(parents=True,exist_ok=True);records={p.parent.name:read(p,{}) for p in root.glob('*/record.json')};used=set()
    def link(path,label):return f'<a href="{esc(path)}">{esc(label)}</a>' if (root/path).is_file() else esc(label)+' (unavailable)'
    def compare(items):
        players=[]
        for path,label in items:
            if not (root/path).is_file():continue
            used.add(path);players.append(f'<figure><video controls playsinline muted preload="metadata" src="{esc(path)}"></video><figcaption>{esc(label)} · {link(path,"MP4")}</figcaption></figure>')
        return '<div class="controls"><button type="button">Play together</button><input aria-label="Shared video time" type="range" min="0" max="1" step="0.0416667" value="0"><output>0 s</output></div><div class="players">'+''.join(players)+'</div>'
    ledger=read(root/'ledger.json',{});elapsed=((ledger.get('finished_unix') or time.time())-ledger.get('started_unix',time.time()))/60
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3 faster VAE · Metal M4 review</title><style>'+CSS+'</style><body><h1>H3 original-model VAE · Metal M4 Max</h1>',
        '<p>Local M4 Max · 128 GiB unified memory. Independent Metal measurements; CUDA results are not used as Metal evidence. Main gates remain <strong>640×480 / 1344×768, 243 frames</strong>. Every generated inspection clip completes <strong>six denoising steps</strong>; decode-only tests reuse proven states.</p>',
        '<p>Balanced uses original weights in BF16 storage, shape-cached MPSGraph matrix operations and fused feed-forward graphs, D=64 Steel attention with FP16 matrix operands and FP32 softmax/accumulation, and FP32 norms/residual/output. Legacy FP32 remains the default; TAEH3 preview is unchanged. M5 NAX is not selected or qualified.</p>',
        '<p class="note">Float-domain metrics precede codec loss. H.264 comparison videos are silent; generation videos below include audio. Players start muted and share play/pause/seek controls. Unmute only one at a time.</p>',
        f'<p>Campaign: {elapsed:.1f} / 240 minutes, including failed attempts. '+link('ledger.json','Ledger')+' · '+link('gates.json','Frozen gates')+' · '+link('inventory.json','Hardware / source inventory')+' · '+link('audit.json','Closing audit')+'</p>']
    parts.append('<nav aria-label="Comparison cases">'+ ' · '.join('<a href="#'+case+'">'+case+'</a>' for case in ['C0','C1','C2','C3','C4-faces','C4-texture'] if (root/(case+'-quality/metrics.json')).is_file())+'</nav>')
    performance=[];summary={}
    for case in ['C0','C1','C2','C3','C4-faces','C4-texture']:
        for mode,job,label in [('reference',case+'-decode','FP32 reference'),('balanced',case+'-decode','Balanced BF16 / Steel'),('tiny',case+'-tiny-bench','TAEH3 preview')]:
            values=[x for x in lines(root/job/'stdout.log') if x.get('mode')==mode]
            if not values:continue
            first=values[0];warm=[x['decode_seconds'] for x in values if x['repeat']>0]
            row=dict(load_seconds=first['load_seconds'],cold_decode_seconds=first['decode_seconds'],warm_seconds=warm,median_warm=statistics.median(warm) if warm else None,
                tracked_peak_bytes=max(x.get('peak_tensor_bytes',0) for x in values),live_bytes=[x.get('live_tensor_bytes') for x in values])
            summary.setdefault(case,{})[mode]=row
            performance.append([case,label,number(row['load_seconds']),number(row['cold_decode_seconds']),number(row['median_warm']),('–'.join([number(min(warm)),number(max(warm))]) if warm else '—'),number(row['tracked_peak_bytes']/2**30) if row['tracked_peak_bytes'] else '—',link(job+'/record.json','record')])
    parts.append('<section><h2>Complete decoder performance</h2>'+table(['Case','Decoder','Load s','First s','Warm median s','Warm range s','Tracked peak GiB','Evidence'],performance)+'<p class="note">Decode includes tile compute, unpacking, output copy and ordered host stitching. First repeats also write raw RGB; warm repeats discard sink payloads. “First” means the first decode for a new decoder, not a purged operating-system file cache. FP32/balanced order alternates. Tracked peak is per decoder; process/Metal totals during pairs include both decoders. Audio and FFmpeg are excluded here. Unified allocations overlap the process footprint and must not be added to it.</p></section>')
    gates={};wall=[]
    for case in ['C1','C2']:
        data=summary.get(case,{});ref=data.get('reference',{}).get('median_warm');balanced=data.get('balanced',{}).get('median_warm')
        speed=ref/balanced if ref and balanced else None
        gates[case]={'decode_speedup':speed,'decode_pass':speed>=1.5 if speed and case=='C1' else (balanced<=ref*1.03 if ref and balanced else None)}
        refjob=records.get(case+'-final-reference',{});baljob=records.get(case+'-final-balanced',{})
        for mode,record in [('reference',refjob),('balanced',baljob)]:
            if record:
                log=root/(case+'-final-'+mode)/'stderr.log';phases={}
                if log.is_file():
                    for label,value in re.findall(r'h3(?:cli)?: phase duration ([^:\n]+): ([0-9.]+) s',log.read_text(errors='replace')):phases[label]=phases.get(label,0)+float(value)
                wall.append([case,mode,esc(record.get('status')),number(record.get('wall_seconds')),number(phases.get('denoise')),number(phases.get('video VAE decode')),str(record.get('completed_steps','—'))+'/6',number(record['peak_physical_footprint_bytes']/2**30) if record.get('peak_physical_footprint_bytes') else '—',link(case+'-final-'+mode+'/record.json','record')])
        if refjob.get('status')=='pass' and baljob.get('status')=='pass':
            gain=1-baljob['wall_seconds']/refjob['wall_seconds'];gates[case].update(wall_gain=gain,wall_pass=gain>=.05)
        else:gates[case].update(wall_gain=None,wall_pass=None)
    if wall:parts.append('<section><h2>Measured complete six-step rendering</h2>'+table(['Case','Decoder','Status','Total wall s','Denoising s','Video decode s','Steps','Sampled footprint GiB','Evidence'],wall)+'<p class="note">Single complete runs, including startup, denoising, decode/audio and mux. Phase times are application timers; whole wall is sampled by the parent process at roughly one-second resolution. The final state hashes are checked separately. These are not inferred by adding decode time to historical denoising measurements.</p></section>')
    rows=[]
    for case,g in gates.items():
        rows.append([case,number(g['decode_speedup'])+'×' if g['decode_speedup'] else 'pending',esc(str(g['decode_pass'])),number(g['wall_gain']*100)+'%' if g['wall_gain'] is not None else 'not measured',esc(str(g['wall_pass']))])
    parts.append('<section><h2>Performance targets and limits</h2>'+table(['Case','Decode speedup vs FP32','Decode target passed','Whole wall saved','5% wall target passed'],rows)+'<p>C1 targets ≥1.5× decode speed and ≥5% whole-render wall reduction. C2 must not regress by more than 3% in complete decode. A memory gain or passing quality does not turn a failed speed target into a pass. Legacy remains the default.</p></section>')
    quality=[]
    for case in ['C0','C1','C2','C3','C4-faces','C4-texture']:
        folder=case+'-quality';data=read(root/folder/'metrics.json')
        if not data:continue
        quality.append(dict(case=case,gates=data['gates']));passed=all(data['gates'].values())
        provenance='Complete six-step latent state; no additional denoising.'
        if case=='C1':provenance+=' Retained BF16 dense park-scene generation; the matched whole-render piano pair below is a separate timing test.'
        if case=='C2':provenance+=' New piano-scene generation, using unchanged BF16 state / Metal SOL denoising and preview presentation.'
        if case=='C0':provenance='Spatial/time crop of the retained complete six-step C1 state.'
        if case=='C3':provenance='Decode-only repeated-tail expansion of the six-step C2 state to 362 frames; not an independent 362-frame generation.'
        if case.startswith('C4'):provenance='Denoiser-free photo pan with text, encoded by the unchanged original reference-video encoder.'
        parts.append('<section class="comparison" id="'+esc(case)+'"><h2>'+esc(case)+f' · {data["width"]}×{data["height"]} · {data["frames"]} frames</h2><p>'+esc(provenance)+' <span class="'+('pass' if passed else 'fail')+'">Quality '+('PASS' if passed else 'FAIL')+'</span></p>'+compare([(folder+'/reference.mp4','FP32 reference'),(folder+'/balanced.mp4','Metal balanced'),(folder+'/tiny.mp4','TAEH3 preview, unchanged')]))
        parts.append(table(['Decoder vs FP32','PSNR dB','SSIM','LPIPS ↓','Worst frame dB','Worst seam dB','Max RGB8 delta'],[[esc(mode),number(m['psnr']),number(m['ssim'],6),number(m['lpips'],7),number(m['worst_psnr']),number(m.get('seam_worst_psnr')),str(m.get('max_rgb8_difference'))] for mode,m in data['modes'].items()]))
        parts.append('<p>'+link(folder+'/metrics.json','Metrics / input hashes')+' · '+link(folder+'/frames.csv','Every frame / temporal joins')+' · '+link(folder+'/record.json','Command')+'</p>')
        for title,names in [('Worst frame and absolute difference ×20',['reference-worst.png','balanced-worst.png','tiny-worst.png','difference-x20.png']),('Spatial seam crop',['seam-reference.png','seam-balanced.png','seam-tiny.png','seam-difference-x20.png'])]:
            existing=[name for name in names if (root/folder/name).is_file()]
            if existing:parts.append('<details><summary>'+esc(title)+'</summary><div class="stills">'+''.join('<figure><img loading="lazy" src="'+esc(folder+'/'+name)+'"><figcaption>'+esc(name)+'</figcaption></figure>' for name in existing)+'</div></details>')
        parts.append('</section>')
    parts.append('<section><h2>Quality contract</h2><p>Every-frame native-resolution float RGB: PSNR ≥35 dB, Gaussian SSIM ≥0.98, worst frame ≥30 dB, ≥3 dB better than TAEH3, and LPIPS AlexNet v0.1 error ≤80% of TAEH3. Seam and temporal-join diagnostics are retained. MPSGraph controls matrix lowering; this is measured close-reference quality, not a claim of bitwise FP32 arithmetic.</p></section>')
    ane_job='ane-row-split-memory-retest' if (root/'ane-row-split-memory-retest/record.json').is_file() else 'ane-row-split'
    ane=lines(root/ane_job/'stdout.log')
    if ane:parts.append('<section><h2>Bounded GPU + ANE experiment</h2>'+table(['Projection','ANE rows','Repeat','GPU s','Split wall s','ANE packing s','ANE prediction s','Relative L2','Speed pass'],[[esc(x.get('projection')),str(x.get('ane_rows')),str(x.get('repeat','—')),number(x.get('gpu_seconds'),5),number(x.get('split_seconds'),5),number(x.get('pack_seconds'),5),number(x.get('predict_seconds'),5),number(x.get('relative_l2'),7),esc(x.get('speed_pass'))] for x in ane])+'<p>'+link(ane_job+'/record.json','Command / memory')+' · '+link(ane_job+'/stderr.log','Compilation / synchronization log')+'. Packing and synchronization count. A rejected split is not integrated into production.</p></section>')
    if ane:
        cold={(x.get('projection'),x.get('ane_rows')):x for x in lines(root/'ane-row-split/stdout.log')}
        setup=[]
        for x in ane:
            if x.get('repeat')!=0:continue
            original=cold.get((x.get('projection'),x.get('ane_rows')), {})
            setup.append([esc(x.get('projection')),str(x.get('ane_rows')),number(original.get('compile_seconds'),4),number(x.get('compile_seconds'),4),number(x.get('ane_memory_bytes',0)/2**20),str(x.get('ane_matmuls'))])
        parts.append('<section><h2>ANE setup and buffers</h2>'+table(['Projection','ANE rows','Initial compile/load s','Cached retest load s','Reported ANE buffers MiB','ANE matmuls'],setup)+'<p>The retest fixes a reporting field: buffer bytes come from compilation statistics, not prediction statistics. They exclude process/library overhead. '+link('ane-row-split/record.json','Initial run')+' · '+link(ane_job+'/record.json','Retest sampled process memory')+'</p></section>')
    audit=read(root/'audit.json',{})
    candidates=[]
    raw={x['candidate']:x for x in audit.get('raw_tile_quality',[])}
    for folder in sorted(root.glob('final-tile-*')):
        values=lines(folder/'stdout.log');warm=[x['decode_seconds'] for x in values if x.get('repeat',0)>0]
        if not values:continue
        q=raw.get(folder.name,{})
        candidates.append([esc(folder.name.removeprefix('final-tile-')),number(values[0]['decode_seconds']),number(statistics.median(warm)) if warm else '—',number(q.get('relative_l2'),7),link(folder.name+'/record.json','recipe / result')])
    if candidates:parts.append('<section><h2>Final isolated 36-block tile candidates</h2>'+table(['Recipe','Cold s','Warm median s','Raw projection relative L2','Evidence'],candidates)+'<p>Same six-step C1 latent tile and final source. Whole-decoder results above determine performance qualification. The raw projected-tile relative-L2 limit is 0.02. FP16 MPSGraph fails that gate, loses speed and shows growing process footprint across its three repeats; direct FP16 MPS passes quality but loses speed. These alternatives remain diagnostics.</p></section>')
    encoder=[]
    for case in ['image-match','image-max','video-short','video-tail']:
        for mode in ['reference','balanced']:
            name='encoder-'+case+'-'+mode;values=lines(root/name/'stdout.log')
            if values:encoder.append([esc(case),mode,esc(json.dumps(values[-1])),link(name+'/record.json','record')])
    if encoder:parts.append('<section><h2>Unchanged FP32 reference encoder</h2>'+table(['Input','Decoder policy','Encoder measurement','Evidence'],encoder)+'<p>Posterior moments, RNG samples and epsilon payloads are checked bitwise across decoder policies in the closing audit.</p></section>')
    if audit:parts.append('<section><h2>Closing correctness and memory audit</h2><p>'+esc(audit.get('summary',''))+'</p><details><summary>Individual checks</summary>'+table(['Check','Result'],[[esc(x['name']),esc(x['passed'])] for x in audit.get('checks',[])])+'</details><p>'+link('audit.json','Full audit and retained checksums')+'</p></section>')
    paired=[
        ('C1 complete six-step renders','Same prompt, seed, geometry and denoiser; decoder policy changes.',[(f'C1-final-{mode}/output.mp4',label) for mode,label in [('reference','FP32 reference'),('balanced','Metal balanced')]]),
        ('Cached decoder-policy isolation','Each video completes six steps; saved video and audio latents are checked bitwise.',[(f'final-session/{mode}.mp4',label) for mode,label in [('reference','FP32 reference'),('balanced','Metal balanced'),('preview','Unchanged TAEH3')]]),
        ('Cached shape-change / repeat','Two complete six-step 96×64 / 39-frame renders; latents and decoded video payloads are checked bitwise.',[(f'final-session/shape-{mode}.mp4',label) for mode,label in [('change','New shape'),('repeat','Cached repeat')]]),
        ('Exact sampler resume','Reference completes six steps; the resumed render continues a diagnostic step-three checkpoint through step six with the balanced decoder.',[(f'C5-resume/{mode}.mp4',label) for mode,label in [('reference','Uninterrupted reference'),('resumed','Resumed / balanced')]]),
        ('Continuation / decode-only','Six-step 90-frame continuation with 39 protected frames delivers 51 frames. Decode-only reuses its saved state and presentation sidecar.',[(folder+'/output.mp4',label) for folder,label in [('C5-continuation-corrected','Generation'),('C5-continuation-decode-corrected','Decode-only')]])]
    for size in ['match','max']:
        items=[]
        for mode,label in [('reference','FP32 reference'),('balanced','Metal balanced')]:
            folder=f'C5-image-{size}-{mode}'
            if records.get(folder+'-extended',{}).get('status')=='pass':folder+='-extended'
            items.append((folder+'/output.mp4',label))
        paired.append(('Reference image size '+size,'Same image, prompt, seed and six completed steps; saved latents are checked bitwise. Max-image checks use the measured M4 vision timeout; the earlier attempt remains in the ledger.',items))
    for title,note,items in paired:
        available=[(path,label) for path,label in items if (root/path).is_file()]
        if not available:continue
        evidence=sorted({str(pathlib.Path(path).parent) for path,label in available})
        parts.append('<section class="comparison"><h2>'+esc(title)+'</h2><p>'+esc(note)+'</p>'+compare(available)+'<p>'+' · '.join(link(folder+'/record.json',folder+' record') for folder in evidence)+'</p></section>')
    remaining=sorted(str(p.relative_to(root)) for p in root.rglob('*.mp4') if str(p.relative_to(root)) not in used and '/venv/' not in str(p))
    groups={}
    for path in remaining:groups.setdefault(path.split('/')[0],[]).append(path)
    for group,paths in groups.items():
        parts.append('<section class="comparison"><h2>'+esc(group)+'</h2>'+compare([(p,pathlib.Path(p).stem) for p in paths])+'<p>'+link(group+'/record.json','Command / result')+'</p></section>')
    parts.append('<section><h2>Complete experiment ledger</h2><p>All attempts are retained. A command pass means execution completed; qualification uses the separate quality/performance gates. Rejected FP16/MPS variants and ANE experiments do not contribute to the selected backend’s speed claims. Optional tests not admitted before the deadline are marked deferred.</p>'+table(['Job','Command status','Wall s','Sampled footprint GiB','Evidence'],[[esc(name),esc(r.get('status')),number(r.get('wall_seconds')),number(r['peak_physical_footprint_bytes']/2**30) if r.get('peak_physical_footprint_bytes') else '—',link(name+'/record.json','record')+' · '+link(name+'/stdout.log','stdout')+' · '+link(name+'/stderr.log','stderr')] for name,r in sorted(records.items(),key=lambda item:item[1].get('started_unix',0))])+'</section>')
    diagnostics=sorted(set(root.glob('*.log'))|set(root.glob('*.jsonl')))
    if diagnostics:parts.append('<section><h2>Setup and preflight records</h2><p>Includes failed build/operator attempts and earlier candidates. These diagnostic tiles are not full-video qualification. The first isolated FP32 timing used an older one-step state; it is excluded from all six-step comparison clips and final performance gates.</p><ul>'+''.join('<li>'+link(p.name,p.name)+'</li>' for p in diagnostics)+'</ul></section>')
    parts.append('<script>'+JS+'</script></body></html>');(root/'review.html').write_text('\n'.join(parts))
    summary.update(quality=quality,performance_gates=gates);(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (root/'performance.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['case','mode','load_seconds','cold_seconds','warm_seconds','tracked_peak_bytes'])
        for case in ['C0','C1','C2','C3','C4-faces','C4-texture']:
            for mode,value in summary.get(case,{}).items():writer.writerow([case,mode,value['load_seconds'],value['cold_decode_seconds'],value['warm_seconds'],value['tracked_peak_bytes']])
    index='<!doctype html><html lang="en"><meta charset="utf-8"><title>H3 VAE reports</title><style>'+CSS+'</style><body><h1>H3 faster original-model VAE</h1>'+''.join('<p><a href="'+backend+'/review.html">'+label+'</a></p>' for backend,label in [('cuda','CUDA qualification and playback'),('metal','Metal M4 qualification and playback')] if (root.parent/backend/'review.html').exists())+'<p>Independent hardware measurements. TAEH3 preview is unchanged; quality, memory and speed gates are reported separately.</p></body></html>'
    (root.parent/'index.html').write_text(index)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            self.paths.extend(unquote(v) for k,v in attrs if k in ('src','href') and v and not v.startswith(('#','http:','https:')))
    pages=[]
    for page in [root/'review.html',root.parent/'index.html']:
        p=Links();p.feed(page.read_text());missing=[s for s in p.paths if not (page.parent/s).is_file()];assert not missing,missing
        pages.append(dict(page=str(page),checked_local_links=len(p.paths),missing=missing))
    media=[]
    if a.verify_media:
        for path in sorted(used):
            streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_type,width,height,nb_read_frames,duration,r_frame_rate','-of','json',str(root/path)],text=True))['streams'];v=next(x for x in streams if x['codec_type']=='video');assert int(v['nb_read_frames'])>0
            media.append(dict(path=path,streams=streams))
    (root/'report-validation.json').write_text(json.dumps(dict(generator_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),pages=pages,media=media,synchronized_controls=True),indent=2)+'\n')
    print(root/'review.html');print(len(records),'records;',len(used),'videos;',sum(x['checked_local_links'] for x in pages),'links checked')
if __name__=='__main__':main()
