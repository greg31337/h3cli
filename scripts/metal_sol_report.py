#!/usr/bin/env python3
"""Retain the bounded SOL sweep and matched 243-frame B2 diagnostic playback."""
import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
from metal_native_bench import ROOT,sha,sol_policy_check


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True)
    a=p.parse_args();base=a.runs.resolve();out=base/'review'
    if out.exists():p.error('use a fresh report directory')
    records={name:json.loads((base/name/'record.json').read_text()) for name in ('dense-B2','sol-B2')}
    for name,r in records.items():
        if not r['evaluation_contract_pass'] or r['geometry']!=[640,480,243] or r['evaluations']!=2 or r['blocks']!=50 or not r['capture_steps'] or not r['preview_vae']:
            p.error('incomplete B2 diagnostic')
        for filename,key in [('result.h3av','av_sha256'),('result.mp4','media_sha256')]:
            if sha(base/name/filename)!=r[key]:p.error('changed media/state')
        if sha(base/name/'bin/h3cli')!=r['manifest']['binary_sha256']:p.error('changed executable')
        for file,digest in r['step_tensor_sha256'].items():
            if sha(base/name/'steps'/file)!=digest:p.error('changed diagnostic capture')
    dense,sol=records.values()
    if dense['manifest']['binary_sha256']!=sol['manifest']['binary_sha256'] or not sol_policy_check(sol) or not sol['sol_performance_evidence']:
        p.error('incompatible binary or missing actual SOL work')
    for key in ('geometry','seed','prompt','conditioning_sha256_after','weight_precision','sampler'):
        if dense[key]!=sol[key]:p.error('unmatched '+key)
    out.mkdir()
    source=ROOT/'tests/metal_tensor_metrics.c';shutil.copy2(source,out/source.name)
    (ROOT/'bin').mkdir(exist_ok=True)
    subprocess.run(['cc','-O3',str(out/source.name),'-lm','-o',str(ROOT/'bin/metal_tensor_metrics')],check=True)
    report={'version':1,'diagnostic_only':True,'geometry':[640,480,243],'evaluations':2,'preview_vae':True,
            'script_sha256':sha(__file__),'record_sha256':{n:sha(base/n/'record.json') for n in records},
            'limits_sha256':sha(ROOT/'tests/metal_sol_mixed_limits.json'),'tensors':{},'production_qualified':False}
    for step in (1,2):
        for modality in ('video','audio'):
            for tensor in ('input','velocity','latent'):
                name=f'step-{step:03d}-{modality}-{tensor}.f32'
                l=base/'dense-B2/steps'/name;r=base/'sol-B2/steps'/name
                metrics=json.loads(subprocess.run([str(ROOT/'bin/metal_tensor_metrics'),str(l),str(r),'f32'],capture_output=True,text=True,check=True).stdout)
                metrics.update(dense_sha256=sha(l),sol_sha256=sha(r),byte_identical=sha(l)==sha(r));report['tensors'][name]=metrics
    if not all(v['byte_identical'] for n,v in report['tensors'].items() if n.startswith('step-001-') or n.endswith('input.f32')):
        p.error('dense protected evaluation or matched routed input differs')
    report['dense_first_and_matched_second_input_pass']=True
    report['routes']={k:sum(r[k] for r in sol['sol']) for k in ('exact','approximate','protected','local')}
    report['actual_exact_fraction']=report['routes']['exact']/(report['routes']['exact']+report['routes']['approximate'])
    report['sol_routed_blocks']=sol['sol_routed_blocks']
    report['invalid_heads']=sum(x['invalid_heads'] for r in records.values() for x in r['mixed'])
    report['timing_diagnostic']={name:[s['wall_seconds'] for s in r['steps']] for name,r in records.items()}
    report['second_evaluation_speedup_diagnostic']=dense['steps'][1]['wall_seconds']/sol['steps'][1]['wall_seconds']
    report['tracked_peak_gib']={name:max(s['metal_peak_bytes'] for s in r['steps'])/2**30 for name,r in records.items()}
    sweep=json.loads((base/'validation/index.json').read_text())
    report['component_screen_pass']=sweep['component_screen_pass']
    report['sweep_sha256']=sha(base/'validation/index.json')
    report['disposition']='Implementation validated. No fixed sweep setting passed every preliminary component speed/error screen. B5/B6 promotion and held-out renders remain deferred; retain dense as the accepted baseline.'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    rows=['<!doctype html><meta charset="utf-8"><title>Conservative Metal SOL — 243 frames</title>',
          '<style>body{font:16px system-ui;background:#161920;color:#eee;max-width:1300px;margin:2rem auto}video{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:1rem}a{color:#9cd3ff}td,th{padding:.5rem;text-align:right}pre{white-space:pre-wrap}</style>',
          '<h1>Conservative SOL: 243-frame diagnostic</h1><p>640×480; two evaluations; preview VAE. This verifies integration and is not B6 production-quality qualification.</p>',
          '<div class="pair"><section><h2>Dense mixed FP16</h2><video id="dense" controls preload="metadata" src="../dense-B2/result.mp4"></video></section><section><h2>SOL: minimum 75% exact</h2><video id="sol" controls muted preload="metadata" src="../sol-B2/result.mp4"></video></section></div>',
          '<button id="play">Play both</button> <button id="pause">Pause both</button> <button id="reset">Restart both</button> <button id="audio-dense">Dense audio</button> <button id="audio-sol">SOL audio</button>',
          '<p>The first evaluation stays dense. The second keeps the first layer, conditioning and recovery heads exact, then routes the remaining blocks. No human judgment is asserted.</p>',
          '<p>'+html.escape(report['disposition'])+'</p><p><a href="report.json">Diagnostic measurements</a> · <a href="../../metal-native-m2-243/continued/accepted/review.html">Accepted M2 production-VAE B6</a></p>',
          '<h2>Standalone real QKV sweep</h2><p>Three warmed alternating timing pairs. Synthetic protection metadata over saved real QKV; the integrated run uses actual model protection.</p>',
          '<table><tr><th>Layer</th><th>Minimum exact</th><th>Speedup</th><th>Relative L2</th><th>Preliminary screen</th></tr>']
    for x in sweep['records']:
        if 'speedup' not in x:continue
        metric=x['results'][0];layer=x['name'].split('-')[0]
        rows.append(f'<tr><td>{layer}</td><td>{metric["min_exact"]:.0%}</td><td>{x["speedup"]:.3f}×</td><td>{metric["relative_l2"]:.6f}</td><td>{"PASS" if x["component_screen_pass"] else "FAIL"}</td></tr>')
    rows+=['</table><h2>Integration record</h2><pre>'+html.escape(json.dumps(report,indent=2))+'</pre>',
           '<script>const videos=[document.querySelector("#dense"),document.querySelector("#sol")];document.querySelector("#play").onclick=()=>{videos[1].currentTime=videos[0].currentTime;videos.forEach(v=>v.play());};document.querySelector("#pause").onclick=()=>videos.forEach(v=>v.pause());document.querySelector("#reset").onclick=()=>videos.forEach(v=>{v.pause();v.currentTime=0;});document.querySelector("#audio-dense").onclick=()=>{videos[0].muted=false;videos[1].muted=true;};document.querySelector("#audio-sol").onclick=()=>{videos[0].muted=true;videos[1].muted=false;};</script>']
    (out/'review.html').write_text('\n'.join(rows)+'\n');print(out/'review.html')

if __name__=='__main__':main()
