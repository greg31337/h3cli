#!/usr/bin/env python3
"""Measure existing adaptive/quantized videos and publish two synchronized triplets."""
import argparse
import csv
import html
from html.parser import HTMLParser
import importlib.metadata
import json
import os
from pathlib import Path
import time
from collections import Counter
from cuda_reference_regression import sha, write
from cuda_adaptive_subblock import media, av_state
from cuda_adaptive_quant import CASES, IDS


def validate_links(out, groups, pairs):
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for name, target in attrs:
                if name in ('href', 'src') and target:
                    assert (out/target).is_file(), target
    Links().feed((out/'index.html').read_text())
    for group in groups:
        assert all((out/p).is_file() for p in group['videos'])
    for pair in pairs:
        assert all((out/p).is_file() for p in pair['worst_images'])


def publish(out, rows, pairs, identity, metric_identity, consistency, *, subblock=False):
    groups = [dict(name=precision.upper(), ids=[r['id'] for r in rows if r['group'] == precision],
                   videos=[r['directory']+'/video.mp4' for r in rows if r['group'] == precision])
              for precision in ('fp8', 'nvfp4')]
    assert all(len(g['videos']) == 3 for g in groups)
    report = dict(schema=1, workload=dict(width=640,height=480,frames=90,steps=50,fps=24,seed=42),
                  human_review='pending', default_changes='none', weight_mode='resident',
                  identity=identity, metrics_identity=metric_identity, variants=rows, comparisons=pairs,
                  dense_consistency=consistency,
                  limitations=['One video per row; single observations, no confidence intervals.',
                               ('SubBlock 0.75 and quantization-only controls both quantize all 50 blocks; adaptive cache is off.' if subblock else
                                'Combined variants keep block 0 BF16; quantization-only controls quantize all 50 blocks.'),
                               'Resident weights throughout; ratios are not comparable to the earlier streamed-weight campaign.',
                               'Packed-weight preparation is separate; loading and all inference/delivery stages are timed.',
                               'Historical similarity diagnostics measure differences and are not a parity or human acceptance gate.',
                               'Earlier comparison approvals do not cover these new videos.'])
    if subblock:report['subblock_sparsity']=.75
    write(out/'report.json', report)
    fields = ['id','wall_seconds','speed_ratio','denoise_seconds','hits','blocks','quantized_calls','peak_vram_gib',
              'peak_rss_gib','ssim_min','lpips_max','audio_relative_l2','historical_similarity_pass']
    if subblock:fields+=['sparse_calls','density','attention_seconds','router_seconds']
    with (out/'report.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    table=''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.4g}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in fields)+'</tr>' for r in rows)
    links=''.join(f'<li>{r["id"]}: <a href="{r["directory"]}/video.mp4">video</a> · <a href="{r["directory"]}/final.h3av">AV state</a> · <a href="{r["directory"]}/result.json">command/timing</a> · <a href="{r["directory"]}/steps.json">step trace</a> · <a href="{r["directory"]}/cache.json">cache decisions</a> · <a href="{r["directory"]}/ffprobe.json">AV timing</a></li>' for r in rows)
    pair_links=''.join(f'<li><a href="metrics/{p["id"]}/result.json">{p["candidate"]} versus {p["reference"]}</a>: minimum SSIM {p["summary"]["ssim_min"]:.4f}, maximum LPIPS {p["summary"]["lpips_max"]:.4f}, audio relative L2 {p["audio"]["relative_l2"]:.4f}</li>' for p in pairs)
    document='''<!doctype html><html><head><meta charset="utf-8"><title>h3cli __TITLE__</title><style>
body{font:16px system-ui;margin:24px;background:#101820;color:#e9eff5}a{color:#82caff}.players{display:flex;gap:12px}.players>div{width:33.33%}video{width:100%;background:#000}table{border-collapse:collapse;display:block;overflow:auto;font-size:13px}td,th{padding:8px;border:1px solid #425564;text-align:right}select,button{font:inherit;padding:6px;margin:10px 5px}img{max-width:100%}.muted{color:#b9c9d6}</style></head><body>
<h1>__TITLE__</h1>
<p>640×480 · 90 frames · 50 steps · 24 FPS · seed 42. Six videos, one per row. Human review: <b>pending</b>. Defaults remain unchanged.</p>
<p class="muted">All weights are resident. Packed-weight preparation is outside timing; load, encode, denoise, full AV decode and delivery are included. Ratios use each triplet's own fresh BF16 baseline. These are single observations.</p>
<label>Triplet <select id="group" onchange="chooseGroup(+this.value)"><option value="0">FP8</option><option value="1">NVFP4</option></select></label>
<button onclick="playAll()">Play all</button><button onclick="pauseAll()">Pause all</button><label>Audio <select id="audio" onchange="chooseAudio()"><option value="2">__COMBINED_SHORT__</option><option value="1">quantization only</option><option value="0">dense BF16</option><option value="-1">muted</option></select></label>
<div class="players"><div><p>Dense BF16</p><video id="v0" controls preload="metadata" muted></video></div><div><p>Quantization only (all 50 blocks)</p><video id="v1" controls preload="metadata" muted></video></div><div><p>__COMBINED_LABEL__</p><video id="v2" controls preload="metadata"></video></div></div>
<h2>Measured results</h2><table><thead><tr>__HEADERS__</tr></thead><tbody>__TABLE__</tbody></table>
<p>Historical similarity checks are retained unchanged. They do not establish perceptual acceptance. __COMPOSITION_NOTE__</p>
<p><a href="report.json">Complete JSON</a> · <a href="report.csv">CSV</a> · <a href="identity.json">Source/build/runtime and preparation</a> · <a href="ledger.json">All attempts</a></p>
<h2>Quality comparisons</h2><ul>__PAIR_LINKS__</ul>
<label>Worst frames <select id="pair" onchange="choosePair(+this.value)">__PAIR_OPTIONS__</select></label><p>Reference, candidate, absolute RGB difference ×10. Frames are selected by lowest SSIM.</p><div id="worst"></div>
<h2>Artifacts</h2><ul>__LINKS__</ul>
<script>const groups=__GROUPS__,pairs=__PAIRS__;const videos=[0,1,2].map(i=>document.getElementById('v'+i));
function pauseAll(){videos.forEach(v=>{if(!v.paused)v.pause()})}function playAll(){const t=videos[2].currentTime;videos.forEach(v=>{if(Math.abs(v.currentTime-t)>.04)v.currentTime=t});Promise.all(videos.map(v=>v.play())).catch(console.error)}
function chooseAudio(){const i=+document.getElementById('audio').value;videos.forEach((v,j)=>v.muted=i!==j)}
function chooseGroup(i){pauseAll();groups[i].videos.forEach((p,j)=>videos[j].src=p);chooseAudio()}
function choosePair(i){document.getElementById('worst').innerHTML=pairs[i].worst_images.map(p=>'<img alt="Reference, candidate and amplified difference" src="'+p+'">').join('')}
videos.forEach(v=>{v.addEventListener('seeking',()=>videos.forEach(w=>{if(w!==v&&Math.abs(w.currentTime-v.currentTime)>.04)w.currentTime=v.currentTime}));v.addEventListener('pause',pauseAll);v.addEventListener('play',()=>videos.forEach(w=>{if(w.paused)w.play().catch(console.error)}));v.addEventListener('ratechange',()=>videos.forEach(w=>w.playbackRate=v.playbackRate))});
setInterval(()=>{if(!videos[2].paused)videos.slice(0,2).forEach(v=>{if(Math.abs(v.currentTime-videos[2].currentTime)>.12)v.currentTime=videos[2].currentTime})},100);chooseGroup(0);choosePair(0);</script></body></html>'''
    substitutions={'__HEADERS__':''.join('<th>'+k+'</th>' for k in fields),'__TABLE__':table,'__LINKS__':links,
                   '__TITLE__':'SubBlock 0.75 with FP8 and NVFP4' if subblock else 'Conservative adaptive cache with FP8 and NVFP4',
                   '__COMBINED_SHORT__':'SubBlock + quantized' if subblock else 'adaptive + quantized',
                   '__COMBINED_LABEL__':'SubBlock 0.75 + quantization (all 50 blocks)' if subblock else 'Conservative cache + quantization (BF16 block 0)',
                   '__COMPOSITION_NOTE__':('Both quantized variants use the same all-50-block projection policy. SubBlock keeps the first ten evaluations, block 0 attention and protected query/key ranges dense; adaptive cache is off.' if subblock else
                                           'Adding caching also restores block 0 to BF16; the combined row is not a pure caching ablation at identical projection precision.'),
                   '__PAIR_LINKS__':pair_links,'__PAIR_OPTIONS__':''.join(f'<option value="{i}">{p["candidate"]} versus {p["reference"]}</option>' for i,p in enumerate(pairs)),
                   '__GROUPS__':json.dumps(groups).replace('<','\\u003c'),'__PAIRS__':json.dumps(pairs).replace('<','\\u003c')}
    for key,value in substitutions.items():document=document.replace(key,value)
    (out/'index.html').write_text(document)
    validate_links(out,groups,pairs)


def main(*, subblock=False):
    if subblock:
        from cuda_subblock_quant import CASES as cases, IDS as ids
    else:cases,ids=CASES,IDS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    out=parser.parse_args().out.resolve()
    env=dict(os.environ)
    assert env.get('CUDA_VISIBLE_DEVICES') == '', 'metrics must run on CPU after rendering'
    ledger=json.loads((out/'ledger.json').read_text())
    assert ledger['complete'] and [c['id'] for c in ledger['cases']]==ids
    identity=json.loads((out/'identity.json').read_text())
    records={};directories={};probes={}
    for case in ledger['cases']:
        d=out/case['artifact_directory'];record=json.loads((d/'result.json').read_text())
        assert record['passed'] and record['manifest_variant']==next(v for v in cases if v['id']==case['id'])
        for name,digest in record['artifacts'].items():assert sha(d/name)==digest
        probes[case['id']]=media(d/'video.mp4',env);av_state(d/'final.h3av')
        records[case['id']]=record;directories[case['id']]=d
    import torch,lpips
    from adaptive_subblock_report import frames,waveform,audio_metrics
    torch.set_num_threads(4)
    net=lpips.LPIPS(net='alex',version='0.1').cpu().eval()
    metric_identity=dict(packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-image','torch','torchvision','lpips','Pillow']},
        weights={p.name:sha(p) for p in (Path(torch.hub.get_dir())/'checkpoints').glob('alexnet-*.pth')}|{'lpips-alex-v0.1':sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')},
        metric_math_sha256=sha(Path(__file__).with_name('adaptive_subblock_report.py')),report_tool_sha256=sha(Path(__file__)))
    if subblock:metric_identity.update(comparison='SubBlock 0.75',entrypoint_sha256=sha(Path(__file__).with_name('subblock_quant_report.py')))
    assert len(metric_identity['weights'])>=2
    audio={i:waveform(d/'video.mp4',env) for i,d in directories.items()}
    thresholds=json.loads((Path(__file__).parent/'cuda_reference/fast-quality.json').read_text())['gates']
    pairs=[]

    def compare(candidate,reference):
        key=candidate+'-vs-'+reference;d=out/'metrics'/key;d.mkdir(parents=True,exist_ok=True)
        path=d/'result.json';candidate_sha=sha(directories[candidate]/'video.mp4');reference_sha=sha(directories[reference]/'video.mp4')
        if path.exists():
            q=json.loads(path.read_text())
            assert q['candidate_sha256']==candidate_sha and q['reference_sha256']==reference_sha and q['metric_identity']==metric_identity
        else:
            started=time.monotonic()
            q=frames(directories[reference]/'video.mp4',directories[candidate]/'video.mp4',d,env,net,torch)
            q.update(audio=audio_metrics(audio[reference],audio[candidate]),candidate_sha256=candidate_sha,
                     reference_sha256=reference_sha,metric_identity=metric_identity,seconds=time.monotonic()-started,human_review='pending')
            q['historical_similarity_checks']=dict(ssim=q['summary']['ssim_min']>=thresholds['every_frame_ssim_min'],
                lpips=q['summary']['lpips_max']<=thresholds['every_frame_lpips_alex_v01_max'],temporal=q['summary']['temporal_max']<=thresholds['temporal_error_rms_max'],
                audio=q['audio']['relative_l2']<=thresholds['audio_relative_l2_max'] and q['audio']['cosine']>=thresholds['audio_cosine_min'])
            write(path,q)
        pair=dict(id=key,candidate=candidate,reference=reference,summary=q['summary'],audio=q['audio'],historical_similarity_checks=q['historical_similarity_checks'],
                  worst_images=[str((d/n).relative_to(out)) for n in q['worst_images']])
        pairs.append(pair)
        print(json.dumps(dict(pair=key,ssim_min=q['summary']['ssim_min'],lpips_max=q['summary']['lpips_max'],audio_relative_l2=q['audio']['relative_l2'])),flush=True)
        return pair

    rows=[]
    for case in cases:
        i=case['id'];record=records[i];d=directories[i];q=compare(i,case['baseline'])
        steps=json.loads((d/'steps.json').read_text());cache=json.loads((d/'cache.json').read_text())
        streams=probes[i]['streams'];video=next(s for s in streams if s['codec_type']=='video');sound=next(s for s in streams if s['codec_type']=='audio')
        row=dict(id=i,group=case['group'],baseline=case['baseline'],directory=str(d.relative_to(out)),wall_seconds=record['wall_seconds'],
                 speed_ratio=records[case['baseline']]['wall_seconds']/record['wall_seconds'],denoise_seconds=sum(s['wall_seconds'] for s in steps),
                 hits=record['counts']['hits'],blocks=record['counts']['blocks'],quantized_calls=record['mixed_precision']['quantized_calls'],
                 peak_vram_gib=record['peak_vram_bytes']/2**30,peak_rss_gib=record['peak_host_rss_bytes']/2**30,
                 ssim_min=q['summary']['ssim_min'],lpips_max=q['summary']['lpips_max'],audio_relative_l2=q['audio']['relative_l2'],
                 historical_similarity_pass=all(q['historical_similarity_checks'].values()),stage_seconds=record['stage_seconds'],counts=record['counts'],
                 precision=record['mixed_precision'],cache_reasons=dict(Counter(c['reason'] for c in cache)),
                 attention_seconds=sum(s['attention_seconds'] for s in steps),h2d_bytes=sum(s['h2d_bytes'] for s in steps),
                 d2h_bytes=sum(s['d2h_bytes'] for s in steps),peak_tensor_bytes=max(s['peak_tensor_bytes'] for s in steps),
                 peak_pinned_bytes=max(s['peak_pinned_bytes'] for s in steps),maximum_sample_gap_seconds=record['maximum_sample_gap_seconds'],
                 missed_samples=record['missed_samples'],video_seconds=float(video['duration']),audio_seconds=float(sound['duration']),
                 av_duration_difference_seconds=float(sound['duration'])-float(video['duration']),quality=q['summary'],audio=q['audio'])
        if subblock:
            possible=record['counts']['possible']
            row.update(sparse_calls=record['counts']['sparse_calls'],density=record['counts']['selected']/possible if possible else 1.,
                       router_seconds=sum(s['router_seconds'] for s in steps),sparse_kernel_seconds=sum(s['sparse_kernel_seconds'] for s in steps),
                       router_calls=sum(s['router_calls'] for s in steps),protected_calls=sum(s['protected_calls'] for s in steps),
                       dense_bypass=sum(s['dense_bypass'] for s in steps),timing_missed=sum(s['attention_timing_missed'] for s in steps))
        if i.endswith('-S75' if subblock else '-A1'):
            control=i.split('-')[0]+'-Q'
            row['versus_quantization_only']=compare(i,control)
            row['quantization_only_wall_ratio']=records[control]['wall_seconds']/record['wall_seconds']
        rows.append(row)
    dense=compare('F4-D0','F8-D0')
    consistency=dict(video_bytes_identical=sha(directories['F4-D0']/'video.mp4')==sha(directories['F8-D0']/'video.mp4'),
                     av_bytes_identical=sha(directories['F4-D0']/'final.h3av')==sha(directories['F8-D0']/'final.h3av'),comparison=dense)
    publish(out,rows,pairs,identity,metric_identity,consistency,subblock=subblock)
    write(out/'assets.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='assets.json'})


if __name__ == '__main__':
    main()
