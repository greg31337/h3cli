#!/usr/bin/env python3
# Historical evidence renderer only. This is not a current qualification gate.
"""Build a local playback report from retained runs; never infer parity from exit 0."""
import argparse
import hashlib
import html
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def inspection_page(root, runs, header):
    """Optional, explicit final pairs; all earlier attempts stay in review.html."""
    selection = root / 'inspection-selection.json'
    if not selection.exists():
        return
    parts = [header, '<h2>Selected final content comparisons</h2><p><a href="review.html">Complete measurements and retained attempts</a> · <a href="fast-regression.html">Protected fast regression</a> · <a href="inspection-selection.json">Selection manifest</a></p><p>Each pair contains complete videos. Matching content does not establish the separate speed or memory gates. Only one video in a pair plays audio.</p>']
    indexed = {r['directory']: r for r in runs}
    for pair in json.loads(selection.read_text())['pairs']:
        records = [indexed[pair[e]] for e in ('sglang', 'native')]
        specs = [r['spec'] for r in records]
        if any(specs[0][k] != specs[1][k] for k in ('case', 'frames', 'evaluations')):
            raise ValueError('incompatible inspection pair')
        if [s['engine'] for s in specs] != ['sglang', 'native']:
            raise ValueError('inspection requires oracle and native reference')
        if any(not r['local_media_verified'] for r in records):
            raise ValueError('inspection media are missing or unverified')
        if '--cuda-reference' not in specs[1]['command']:
            raise ValueError('inspection native candidate is not reference mode')
        path = root / pair['metrics']
        if not path.resolve().is_relative_to(root):
            raise ValueError('inspection metrics escape report root')
        metrics = json.loads(path.read_text())
        if [metrics[k] for k in ('reference_sha256', 'candidate_sha256')] != [r['validation']['sha256'] for r in records]:
            raise ValueError('stale inspection metrics')
        s = specs[0]
        parts.append(f'<section><h2>{html.escape(s["case"])} · {s["frames"]} frames / {s["evaluations"]} evaluations</h2><p>Decoded video/audio gates: '+('PASS' if metrics['passed'] else 'FAIL')+'</p><div class="cards">')
        for label, r in zip(('SGLang oracle', 'Native reference'), records):
            directory = html.escape(r['directory'])
            parts.append(f'<article><h3>{label}</h3><video controls preload="none" src="{directory}/video.mp4"></video><p><a href="{directory}/command.json">Command and identity</a> · <a href="{directory}/result.json">Run record</a></p></article>')
        relative = html.escape(pair['metrics'])
        parts.append(f'</div><button onclick="playPair(this)">Play pair from start</button><button onclick="stopPair(this)">Pause pair</button><p><a href="{relative}">Every-frame metrics and audio comparison</a></p>')
        exact = all(f['mse'] == 0 for f in metrics['frames'])
        parts.append('<p>Decoded RGB: '+('bit-identical on every frame' if exact else 'see frame metrics')+'. Audio relative L2: '+str(metrics['audio'].get('relative_l2'))+'.</p>')
        trajectory = root / pair['trajectory']
        if not trajectory.resolve().is_relative_to(root):
            raise ValueError('inspection trajectory escapes report root')
        trace = json.loads(trajectory.read_text())
        if trace.get('case') != s['case'] or trace.get('evaluations') != s['evaluations']:
            raise ValueError('incompatible inspection trajectory')
        parts.append('<p>Complete free-running trajectory: '+('bit-identical' if trace.get('passed') and trace.get('bitwise_equal') else 'see numerical gates')+
                     '. <a href="'+html.escape(pair['trajectory'])+'">Every velocity and sampler update</a>.</p>')
        if s['case'] in ('C0', 'C1', 'C2'):
            parts.append('<p><a href="matched-campaign-v13/result.json">Three-pair timing and VRAM results</a> · <a href="../records/profile-analysis-v12.json">Per-step allocation history</a></p>')
        parts.append('<details><summary>Worst-frame comparison and amplified difference</summary>')
        for name in ('reference-worst.png', 'candidate-worst.png', 'difference-x20.png'):
            image = path.parent / name
            if image.exists():
                parts.append('<img loading="lazy" style="max-width:100%" src="'+html.escape(str(image.relative_to(root)))+'" alt="'+name+'">')
        parts.append('</details></section>')
    parts.append('''<script>
function pairVideos(b){return [...b.closest('section').querySelectorAll('video')]}
function playPair(b){pairVideos(b).forEach((v,i)=>{v.currentTime=0;v.muted=i>0;v.play().catch(()=>{})})}
function stopPair(b){pairVideos(b).forEach(v=>v.pause())}
document.querySelectorAll('video').forEach(v=>v.addEventListener('seeked',()=>{
 if(v.dataset.sync==='1'){v.dataset.sync='0';return}
 pairVideos(v).forEach(w=>{if(w!==v&&Math.abs(w.currentTime-v.currentTime)>.075){w.dataset.sync='1';w.currentTime=v.currentTime}})
}));
</script></html>''')
    (root / 'final-review.html').write_text(''.join(parts))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);a=p.parse_args()
    root=a.root.resolve();runs=[];checksums={}
    for path in sorted(root.rglob("result.json")):
        result=json.loads(path.read_text())
        if "spec" not in result:continue
        result["directory"]=str(path.parent.relative_to(root));result["id"]=result["directory"]
        media=path.parent/"video.mp4";valid=result.get("validation",{}).get("sha256")
        if media.exists() and valid and digest(media)!=valid:raise ValueError(f"changed media: {media}")
        result["local_media_verified"]=bool(valid and media.exists());runs.append(result)
    header='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CUDA BF16 / SGLang parity</title><style>
body{font:16px system-ui;background:#15171b;color:#eee;max-width:1500px;margin:25px auto;padding:20px}a{color:#9bd0ff}h1,h2{line-height:1.2}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:20px}article,section{background:#23262c;padding:18px;margin:16px 0;border-radius:8px}article{background:#30343b}video{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}td,th{text-align:left;padding:7px;border-bottom:1px solid #555}table{border-collapse:collapse;width:100%}.pending{color:#ffd37b}.pass{color:#b2f6bc}button{padding:8px;margin:5px}small{color:#c7ccd3}
</style><h1>CUDA BF16 reference parity with SGLang</h1>
<p class="pending">Qualification in progress. Valid media and oracle repeatability do not establish native content, speed or memory parity. Candidates are labeled separately from the protected fast baseline.</p>
<p>640×480, 24 fps, original FL2VA/Ref2VA weights and full VAE. C0: 124 frames / 6 evaluations; C1: 362 frames / 6 evaluations; C2: 124 frames / 50 evaluations. SGLang uses 7 / 7 / 51 sigma-grid points.</p>
<p><a href="report-data.json">All retained results</a> · <a href="report-checksums.json">Local asset checksums</a></p>'''
    qualification=root/'qualification.json'
    if qualification.exists():
        final=json.loads(qualification.read_text())
        header=header.replace('Qualification in progress. Valid media and oracle repeatability do not establish native content, speed or memory parity.',
                              'Qualification: '+html.escape(final['status'])+'.')
        header+='<p><a href="qualification.json">Frozen-gate closeout and remaining failures</a></p><table><tr><th>Gate</th><th>Result</th></tr>'
        for name,passed in final['gates'].items():
            label='ACCEPTED BY USER; measured exceptions retained' if name in final.get('waived_gates',{}) else 'PASS' if passed else 'FAIL / OPEN'
            header+='<tr><td>'+html.escape(name)+'</td><td>'+label+'</td></tr>'
        header+='</table>'
        if final.get('waived_gates'):
            header+='<p><a href="qualification-waivers.json">Recorded user waiver</a>. Measured differences remain in the report.</p>'
    inspection_page(root,runs,header)
    parts=[header]
    if (root/'final-review.html').exists():
        parts.append('<p><a href="final-review.html">Open selected final pairs for visual inspection</a></p>')
    def link(directory,name,label=None):
        path=root/directory/name
        return f'<a href="{html.escape(str(path.relative_to(root)))}">{html.escape(label or name)}</a>' if path.exists() else ""
    parts.append('<p><a href="fast-regression.html">Protected fast CUDA baseline and regression evidence</a></p>')
    archive_records=('build-identity-final-v25.json','reference-final-delta-v25.json',
                     'reference-source-v26.json','reference-source-v26.tar.gz',
                     'source-stability-v26.json','content-summary-v26.json')
    available=[name for name in archive_records if (root.parent/'records'/name).is_file()]
    if available:
        parts.append('<details><summary>Source/build identities and final evidence summaries</summary><p>')
        for name in available:
            relative='../records/'+name
            checksums[relative]=digest(root.parent/'records'/name)
            parts.append('<a href="'+relative+'">'+html.escape(name)+'</a><br>')
        parts.append('</p></details>')
    supplementary=[]
    for path in sorted(root.rglob('result.json')):
        record=json.loads(path.read_text())
        kind=record.get('comparison',record.get('kind',''))
        if kind not in ('teacher_forced_every_evaluation','native reference interruption and cache replay',
                        'matched fresh-process CUDA reference campaign','protected fast before/after, production defaults'):
            continue
        directory=str(path.parent.relative_to(root))
        supplementary.append((directory,kind,record))
    if supplementary:
        parts.append('<section><h2>Campaign gates and replay checks</h2><p>These checks answer different questions. Teacher-forced replay, interrupted replay and fresh-process performance are kept separate.</p><table><tr><th>Evidence</th><th>Scope</th><th>Status</th></tr>')
        for directory,kind,record in supplementary:
            parts.append('<tr><td>'+link(directory,'result.json',directory)+'</td><td>'+html.escape(kind)+'</td><td class="'+('pass' if record.get('passed') else 'pending')+'">'+('PASS' if record.get('passed') else 'FAIL / INCOMPLETE')+'</td></tr>')
        parts.append('</table>')
        for directory,kind,record in supplementary:
            if kind=='matched fresh-process CUDA reference campaign':
                parts.append('<h3>'+html.escape(directory)+'</h3><table><tr><th>Case</th><th>Measurement</th><th>SGLang median [range]</th><th>Native median [range]</th><th>Native / oracle</th><th>Gate</th></tr>')
                for case,summary in record.get('summary',{}).items():
                    for name,m in summary.get('measurements',{}).items():
                        if not m.get('complete'):continue
                        factor=2**30 if name=='peak_vram' else 1
                        unit='GiB' if name=='peak_vram' else 's'
                        cells=[case,name+' ('+unit+')']
                        for engine in ('sglang','native'):
                            v=m[engine];cells.append(f"{v['median']/factor:.3f} [{v['minimum']/factor:.3f}, {v['maximum']/factor:.3f}]")
                        cells += [f"{m['native_over_oracle']:.3f}",('PASS' if m['passed'] else 'FAIL') if 'passed' in m else 'Reported separately']
                        parts.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>')
                parts.append('</table><p>Generation retains each engine’s loading boundary. The native value includes lazy weights. A sampling-gap failure leaves peak-memory qualification open even when the sampled native peak is lower.</p>')
            if 'summary' in record:
                parts.append('<details><summary>'+html.escape(directory)+' — measured gates</summary><pre>'+html.escape(json.dumps(record['summary'],indent=2))+'</pre></details>')
            if kind=='native reference interruption and cache replay':
                parts.append('<details><summary>'+html.escape(directory)+' — completed replay videos</summary><div class="cards">')
                for row in record.get('records',[]):
                    media=root/directory/row['name']/'video.mp4'
                    if not media.exists():continue
                    expected=row.get('validation',{}).get('sha256')
                    if not expected or digest(media)!=expected:raise ValueError('unverified resume media: '+str(media))
                    parts.append('<article><p>'+html.escape(row['name'])+'</p><video controls preload="metadata" src="'+html.escape(str(media.relative_to(root)))+'"></video></article>')
                parts.append('</div></details>')
        parts.append('</section>')
    profiles=root.parent/'records/profile-analysis-v12.json'
    if profiles.exists():
        profile=json.loads(profiles.read_text())
        relative='../records/'+profiles.name
        checksums[relative]=digest(profiles)
        parts.append('<section><h2>Memory and timing through every evaluation</h2><p>Instrumented native diagnostics, excluded from latency qualification. Transfers overlap computation; category times must not be added together. Native tensor counters exclude CUDA driver and library overhead.</p><p><a href="'+relative+'">Complete profiles and allocation histories</a></p>')
        for case in profile['cases']:
            parts.append('<details><summary>'+html.escape(case['case'])+' — '+str(len(case['steps']))+' evaluations; allocation plateau '+('PASS' if case['live_allocation_plateau'] else 'FAIL')+'</summary><table><tr><th>Evaluation</th><th>Wall s</th><th>GEMM s</th><th>Attention s</th><th>Exposed upload wait s</th><th>Live device GiB</th><th>Pinned GiB</th><th>RSS GiB</th><th>Swap GiB</th></tr>')
            for row in case['steps']:
                cells=[str(row['step'])]+[f"{row[k]:.3f}" for k in ('wall_seconds','gemm_seconds','attention_seconds','upload_wait_seconds')]+[f"{row[k]/2**30:.3f}" for k in ('live_device_bytes','pinned_bytes','resident_bytes','swap_used_bytes')]
                parts.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
            parts.append('</table></details>')
        parts.append('</section>')
    audio_root=root.parent/'audio-campaign-v1'
    if (audio_root/'result.json').exists():
        campaign=json.loads((audio_root/'result.json').read_text())
        parts.append('<section><h2>Crossed audio decoder replay</h2><p>Updated decoder on retained clean latents. These results do not replace the older complete renders below or qualify generation speed.</p><div class="cards">')
        for row in campaign['results']:
            case=row['case'];folder=audio_root/case
            parts.append('<article><h3>'+html.escape(case)+'</h3><p class="'+('pass' if row['passed'] else 'pending')+'">Decoded audio gate: '+('PASS' if row['passed'] else 'FAIL')+'</p>')
            for name in ('raw_pcm','encoded_pcm'):
                m=row.get(name,{})
                parts.append('<p>'+html.escape(name)+': '+('bit-for-bit identical' if m.get('exact') else 'relative L2 '+html.escape(str(m.get('relative_l2'))))+'</p>')
            for engine in ('oracle','native'):
                path=folder/engine/'encoded.m4a';relative='../audio-campaign-v1/'+case+'/'+engine+'/encoded.m4a'
                if path.exists():
                    actual=digest(path)
                    if actual!=row['sha256'][engine+'/encoded.m4a']:raise ValueError('changed audio replay: '+str(path))
                    checksums[relative]=actual
                    parts.append('<p>'+engine+'</p><audio controls preload="metadata" src="'+html.escape(relative)+'"></audio>')
            relative='../audio-campaign-v1/'+case+'/result.json';checksums[relative]=digest(folder/'result.json')
            parts.append('<p><a href="'+html.escape(relative)+'">Metrics and retained commands</a></p></article>')
        parts.append('</div></section>')
    video_root=root.parent/'vae-campaign-v2'
    if (video_root/'result.json').exists():
        campaign=json.loads((video_root/'result.json').read_text())
        parts.append('<section><h2>Crossed full-video decoder replay</h2><p>Independent native and oracle clean AV payloads are checked first. Identical payloads are decoded once by each implementation, covering all four crossed combinations. Raw RGB checks include every frame; generation speed remains a separate gate.</p><table><tr><th>Case</th><th>Raw RGB</th><th>Records</th></tr>')
        for row in campaign['results']:
            relative='../vae-campaign-v2/'+row['case']+'/result.json'
            checksums[relative]=digest(video_root/row['case']/'result.json')
            parts.append('<tr><td>'+html.escape(row['case'])+'</td><td>'+('Bit-for-bit identical' if row.get('raw_rgb',{}).get('exact') else 'FAILED / incomplete')+'</td><td><a href="'+relative+'">Checksums and metrics</a></td></tr>')
        parts.append('</table></section>')
    cases=list(dict.fromkeys(['C0','C1','C2']+[r['spec']['case'] for r in runs]))
    for case in cases:
        group=[r for r in runs if r["spec"]["case"]==case and r['spec']['engine']!='fast']
        parts.append(f'<section><h2>{case}</h2><p class="pending">Content / latency / VRAM / fast isolation: see individual evidence; unmeasured gates remain open.</p>')
        completed=[r for r in group if r["spec"]["engine"]=="sglang" and r["local_media_verified"] and not r["spec"].get("instrumented")]
        if len(completed)>=2:
            same=len({r["validation"]["sha256"] for r in completed})==1
            parts.append(f'<p>Oracle repeatability: {"byte-identical MP4s" if same else "MP4s differ; numerical comparison required"} across {len(completed)} retained runs.</p>')
        parts.append('<div class="cards">')
        for r in group:
            spec=r["spec"];d=r["directory"];v=r.get("validation",{});candidate="--cuda-reference" in spec["command"]
            name="SGLang oracle" if spec["engine"]=="sglang" else "Native reference" if candidate else "Protected fast baseline" if spec["engine"]=="fast" else "Legacy native baseline"
            parts.append(f'<article><h3>{name}</h3><p>{html.escape(r["id"])} · {html.escape(r["status"])} · {spec["frames"]} frames / {spec["evaluations"]} evaluations</p>')
            if r["local_media_verified"]:parts.append(f'<video controls preload="metadata" src="{html.escape(d)}/video.mp4"></video>')
            if spec.get("instrumented"):parts.append('<p class="pending">Instrumented diagnostic run. Excluded from performance qualification.</p>')
            if any(v.startswith('H3_TEST_NATIVE_TEACHER_DIR=') and v.split('=',1)[1] for v in spec.get('overrides',[])):
                parts.append('<p class="pending">Teacher-forced diagnostic. Oracle states are supplied between evaluations; this does not establish ordinary free-running generation.</p>')
            if any(v.startswith(('H3_TEST_SGLANG_INPUT_DIR=','H3_TEST_SGLANG_CONDITION_DIR=')) and v.split('=',1)[1] for v in spec.get('overrides',[])):
                parts.append('<p class="pending">Oracle conditioning imported. This diagnostic does not qualify native conditioning or ordinary same-seed generation.</p>')
            measurements={"Process wall (s)":r.get("wall_seconds"),"Verified complete playable output (s)":r.get("complete_playable_seconds"),"Generation (s)":v.get("generation_seconds"),"Peak total GPU (GiB)":r.get("sampled_peak_gpu_bytes",0)/2**30,"Largest sampling gap (ms)":r.get("largest_sample_gap_seconds",0)*1000}
            parts.append('<table>'+''.join(f'<tr><th>{html.escape(k)}</th><td>{x:.3f}</td></tr>' for k,x in measurements.items() if x is not None)+'</table>')
            stages=v.get('stage_seconds',{})
            if stages:
                parts.append('<details><summary>Every measured stage, including native model loading and delivery</summary><table>')
                parts.extend('<tr><th>'+html.escape(k)+'</th><td>'+f'{seconds:.6f} s'+'</td></tr>' for k,seconds in stages.items())
                parts.append('</table>')
                if all(k in stages for k in ('audio VAE','video VAE load','video VAE decode','FFmpeg')):
                    total=sum(stages[k] for k in ('audio VAE','video VAE load','video VAE decode','FFmpeg'))
                    parts.append(f'<p>Native AV decode including video-model loading and delivery: {total:.3f} s. All of this work also remains in generation and process wall time.</p>')
                parts.append('</details>')
            parts.append('<p>'+' · '.join(filter(None,(link(d,"command.json"),link(d,"result.json"),link(d,"render.log"),link(d,"performance.json"),link(d,"ffprobe.json"))))+'</p>')
            metrics=next((root/d/name for name in ('media-metrics-v2/metrics.json','media-metrics/metrics.json') if (root/d/name).exists()),root/d/'media-metrics/metrics.json')
            if metrics.exists():
                m=json.loads(metrics.read_text())
                if m['candidate_sha256']!=v.get('sha256'):raise ValueError(f'stale media scores: {metrics}')
                parts.append('<p class="'+('pass' if m['passed'] else 'pending')+'">Decoded content gates: '+('PASS' if m['passed'] else 'FAIL')+'</p>')
                parts.append('<p>'+link(d,str(metrics.relative_to(root/d)),"Every-frame video and audio metrics")+'</p>')
                parts.append('<table>'+''.join('<tr><th>'+html.escape(k)+'</th><td>'+('PASS' if v else 'FAIL')+'</td></tr>' for k,v in m['gates'].items())+'</table>')
                frame=m['worst_frame']
                parts.append(f'<details><summary>Worst frame {frame}: reference, candidate, absolute difference ×20</summary>')
                for name in ('reference-worst.png','candidate-worst.png','difference-x20.png'):
                    image_path=metrics.parent/name
                    if image_path.exists():parts.append(f'<img loading="lazy" style="width:100%" alt="{html.escape(name)}" src="{html.escape(str(image_path.relative_to(root)))}">')
                parts.append('</details>')
                scores=m['frames'];minimum=min(x['psnr_db'] for x in scores);maximum=min(65,max(41,max(x['psnr_db'] for x in scores)))
                low=min(35,minimum);span=max(1,maximum-low)
                points=' '.join(f'{20+460*i/max(1,len(scores)-1):.2f},{130-110*(min(x["psnr_db"],maximum)-low)/span:.2f}' for i,x in enumerate(scores))
                gate_y=130-110*(40-low)/span
                parts.append(f'<svg viewBox="0 0 500 160" role="img" aria-label="PSNR across every frame, dashed line is 40 dB gate"><path d="M20 {gate_y:.2f}H480" stroke="#ffd37b" stroke-dasharray="5 4"/><polyline points="{points}" fill="none" stroke="#9bd0ff"/><text x="20" y="155" fill="white">Every-frame PSNR ({minimum:.2f} dB minimum); gate 40 dB</text></svg>')
            boundary=next((root/d/name for name in ('trajectory-v2.json','trajectory.json','inputs.json','boundaries.json','boundaries-early.json') if (root/d/name).exists()),None)
            if boundary:
                b=json.loads(boundary.read_text())
                parts.append('<p>First failing captured boundary: '+html.escape(str(b.get('first_failing_boundary')))+' · '+html.escape(b.get('comparison','conditioning input comparison'))+'</p><p>'+link(d,boundary.name,'Tensor comparison')+'</p>')
                trajectory=[v for v in b['results'] if v.get('kind')=='trajectory']
                if trajectory:
                    parts.append('<p class="'+('pass' if b.get('passed') else 'pending')+'">Free-running trajectory: '+('bit-for-bit identical' if b.get('bitwise_equal') else 'see numerical gates')+'. Final decoder/content qualification is separate.</p>')
                    parts.append('<details><summary>Every retained sampler update</summary><table><tr><th>Boundary</th><th>Relative L2</th><th>Maximum error</th><th>Gate</th></tr>')
                    for v in trajectory:
                        parts.append('<tr><td>'+html.escape(v['name'])+'</td><td>'+html.escape(str(v.get('relative_l2','missing')))+'</td><td>'+html.escape(str(v.get('max_abs','missing')))+'</td><td>'+('PASS' if v.get('passed') else 'FAIL')+'</td></tr>')
                    parts.append('</table></details>')
            telemetry=root/d/'telemetry.jsonl'
            if telemetry.exists():
                samples=[json.loads(line) for line in telemetry.read_text().splitlines() if line.strip()]
                if samples:
                    duration=max(1,samples[-1]['elapsed_s']);peak=max(1,max(v['gpu_used_bytes'] for v in samples))
                    # Break the visual at missing samples too. A connecting line
                    # across a blocked NVML query would imply unobserved usage.
                    segments=[[]]
                    for v in samples:
                        if segments[-1] and v['elapsed_s']-segments[-1][-1]['elapsed_s']>.1:
                            segments.append([])
                        segments[-1].append(v)
                    paths=[]
                    for segment in segments:
                        stride=max(1,len(segment)//500)
                        drawn=segment[::stride]
                        if drawn[-1] is not segment[-1]:drawn.append(segment[-1])
                        points=' '.join(f'{20+460*v["elapsed_s"]/duration:.2f},{130-110*v["gpu_used_bytes"]/peak:.2f}' for v in drawn)
                        paths.append(f'<polyline points="{points}" fill="none" stroke="#9bd0ff"/>')
                    parts.append(f'<details><summary>Total GPU memory history; {len(samples)} retained samples</summary><svg viewBox="0 0 500 160" role="img" aria-label="Total GPU memory over process time; gaps above 100 ms are left blank">'+''.join(paths)+f'<text x="20" y="155" fill="white">0–{duration:.1f} seconds; sampled peak {peak/2**30:.2f} GiB</text></svg><p>Gaps above 100 ms are left blank. Complete samples are retained; the plot may be downsampled.</p><p>'+link(d,'telemetry.jsonl','Complete memory samples')+'</p></details>')
            parts.append('<details><summary>Exact command</summary><pre>'+html.escape(json.dumps(spec["command"],indent=2))+'</pre></details>')
            if r.get("error"):parts.append('<p class="pending">'+html.escape(r["error"])+"</p>")
            parts.append('</article>')
        if not group:parts.append('<p class="pending">No completed local evidence yet.</p>')
        parts.append('</div><button onclick="playGroup(this)">Play together from start</button><button onclick="pauseGroup(this)">Pause all</button></section>')
    parts.append('''<script>
function videos(b){return [...b.closest('section').querySelectorAll('video')]}
function playGroup(b){videos(b).forEach((v,i)=>{v.currentTime=0;v.muted=i>0;v.play().catch(()=>{})})}
function pauseGroup(b){videos(b).forEach(v=>v.pause())}
document.querySelectorAll('video').forEach(v=>v.addEventListener('seeked',()=>{
 if(v.dataset.seeking==='1'){v.dataset.seeking='0';return}
 videos(v).forEach(w=>{if(w!==v&&Math.abs(w.currentTime-v.currentTime)>.075){w.dataset.seeking='1';w.currentTime=v.currentTime}})
}));
</script></html>''')
    (root/"review.html").write_text(''.join(parts))
    fast=[header,'<h2>Protected fast CUDA</h2><p><a href="review.html">Return to reference qualification</a></p><p>These artifacts measure existing fast behavior. They are excluded from reference-content qualification. Missing post-change comparisons remain open.</p>']
    baseline=root.parent/'records/isolation-v1/baseline.bf16'
    candidate=root.parent/'records/gpu-compat-final-v25/fast.bf16'
    if not candidate.exists():candidate=root.parent/'records/gpu-compat-v12/fast.bf16'
    if baseline.exists() and candidate.exists():
        same=digest(baseline)==digest(candidate)
        fast.append('<p>Source-independent fixed-tuning fixture: '+('bit-for-bit unchanged' if same else 'FAILED: output changed')+'. Candidate also exercises mixed fast/reference contexts and cancellation. Full-render performance and quantization coverage remain separate gates.</p>')
        fast.append('<p><a href="../records/isolation-v1/baseline.log">Baseline log</a> · <a href="../records/'+candidate.parent.name+'/isolation.log">Candidate log</a></p>')
    for directory,kind,record in supplementary:
        if kind=='protected fast before/after, production defaults':
            fast.append('<section><h2>Matched production defaults</h2><p class="'+('pass' if record.get('passed') else 'pending')+'">'+('PASS' if record.get('passed') else 'FAIL / INCOMPLETE')+'</p><p>'+link(directory,'result.json','Before/after commands, checksums and measured gates')+'</p><pre>'+html.escape(json.dumps(record.get('summary',{}),indent=2))+'</pre></section>')
            if (root/directory/'denoising.json').exists():
                denoising=json.loads((root/directory/'denoising.json').read_text())
                fast.append('<p>Protected fast denoising medians: '+('PASS' if denoising['passed'] else 'FAIL')+' · '+link(directory,'denoising.json','All six step durations in each uninstrumented run')+'</p>')
    evidence=('fast-source-audit-v14.json','fast-source-audit-final-v25.json',
              'generator-final-v25.json','initial-rng-final-v25/result.json',
              'gpu-compat-final-v25/result.json','no-cudnn-rejection-final-v25.json',
              'encoder-worker-version-v13.json',
              'reference-session-v14.log','reference-lifetime-v14.log',
              'reference-session-v15.log','reference-lifetime-v15.log',
              'metal-compat-gpu-v14.log','no-cudnn-rejection-v1.json')
    retained=[name for name in evidence if (root.parent/'records'/name).exists()]
    if retained:
        fast.append('<section><h2>Isolation and compatibility checks</h2><p>')
        for name in retained:
            relative='../records/'+name
            checksums[relative]=digest(root.parent/'records'/name)
            fast.append('<a href="'+relative+'">'+html.escape(name)+'</a><br>')
        fast.append('</p></section>')
    for r in runs:
        if r['spec']['engine']!='fast':continue
        d=r['directory'];fast.append('<section><h3>'+html.escape(r['id'])+'</h3>')
        if r['local_media_verified']:fast.append(f'<video controls preload="metadata" src="{html.escape(d)}/video.mp4"></video>')
        fast.append('<p>'+' · '.join(filter(None,[link(d,'command.json'),link(d,'result.json'),link(d,'render.log')]))+'</p></section>')
    fast.append('</html>');(root/'fast-regression.html').write_text(''.join(fast))
    (root/"report-data.json").write_text(json.dumps(runs,indent=2)+"\n")
    allowed={".json",".jsonl",".log",".txt",".mp4",".m4a",".png",".jpg",".html",".csv"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in allowed and path.name!="report-checksums.json":checksums[str(path.relative_to(root))]=digest(path)
    (root/"report-checksums.json").write_text(json.dumps(checksums,indent=2)+"\n")
    print(f"Reported {len(runs)} retained runs in {root/'review.html'}")


if __name__=="__main__":main()
