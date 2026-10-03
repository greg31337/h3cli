#!/usr/bin/env python3
"""Verify M3C evidence, apply existing quality screens, and publish playback pages."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from metal_native_bench import ROOT,sha
from metal_ane_validate import MODES

def conditioning_payloads(record):
    command=record['command'];flag='--load-conditioning' if '--load-conditioning' in command else '--save-conditioning'
    path=Path(command[command.index(flag)+1])
    if sha(path)!=record['conditioning_sha256_after']:raise ValueError('changed conditioning cache')
    data=path.read_bytes()
    if data[:8]!=b'H3COND\0\0' or len(data)<128:raise ValueError('invalid cache')
    count=struct.unpack_from('<I',data,24)[0]
    if not 1<=count<=80:raise ValueError('invalid cache table')
    result={}
    for i in range(count):
        table=data[128+i*96:128+(i+1)*96]
        if len(table)!=96:raise ValueError('truncated cache')
        id=struct.unpack_from('<I',table)[0];start,length=struct.unpack_from('<QQ',table,40)
        if start+length>len(data):raise ValueError('truncated payload')
        if id!=1:result[id]=hashlib.sha256(data[start:start+length]).hexdigest()
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True,help='Retained original MPSGraph B6 directory')
    p.add_argument('--controls',type=Path,required=True,help='Passing reference-only calibration metrics.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();runs=a.runs.resolve();out=a.output.resolve();reference=a.reference.resolve()
    if out.exists():p.error('use a fresh report directory')
    index=json.loads((runs/'index.json').read_text())
    if set(index['runs'])!={f'{case}-{mode}' for case in ('B1','B5','B6') for mode in MODES}:p.error('incomplete suite')
    if sha(runs/'operators.jsonl')!=index['operators_sha256']:p.error('changed operator evidence')
    records={};snapshots=set()
    for name,entry in index['runs'].items():
        path=runs/name
        if sha(path/'record.json')!=entry['record_sha256'] or sha(path/'run.log')!=entry['log_sha256']:p.error('changed record/log')
        r=json.loads((path/'record.json').read_text());records[name]=r
        if not r['evaluation_contract_pass'] or r['geometry']!=[640,480,243] or r['blocks']!=50:p.error('invalid workload')
        if sha(path/'bin/h3cli')!=index['binary_sha256'] or r['manifest']['binary_sha256']!=index['binary_sha256']:p.error('changed binary')
        for file,key in [('result.h3av','av_sha256'),('result.mp4','media_sha256')]:
            if sha(path/file)!=r[key]:p.error('changed output')
        for file,digest in r.get('step_tensor_sha256',{}).items():
            if sha(path/'steps'/file)!=digest:p.error('changed tensor')
        frozen=r['manifest']['binary_build_provenance']
        if frozen['source_snapshot'] not in snapshots:
            for file,digest in frozen['source_sha256'].items():
                if sha(Path(frozen['source_snapshot'])/file)!=digest:p.error('changed frozen source')
            snapshots.add(frozen['source_snapshot'])
        if r['case']=='B5' and (r['capture_steps'] or r['component_fences'] or r['region_fences'] or r.get('trace_command')):p.error('instrumented B5')
        if r['case']=='B6' and (r['evaluations']!=6 or r['preview_vae']):p.error('invalid B6')
    for case in ('B1','B5','B6'):
        base=records[case+'-off']
        for mode in MODES:
            r=records[case+'-'+mode]
            for key in ('geometry','seed','prompt','evaluations','blocks','conditioning_sha256_after','preview_vae','sampler','weight_precision'):
                if r[key]!=base[key]:p.error('unmatched '+key)
            options={k:v for k,v in r['metal_options'].items() if k!='metal-ane'}
            if options!={k:v for k,v in base['metal_options'].items() if k!='metal-ane'}:p.error('changed attention recipe')
    old=json.loads((reference/'record.json').read_text())
    if not old['evaluation_contract_pass'] or old['backend']!='mpsgraph' or old['case']!='B6' or old['evaluations']!=6 or old['preview_vae']:p.error('invalid original reference')
    for key in ('geometry','seed','prompt','blocks','sampler','weight_precision'):
        if old[key]!=records['B6-off'][key]:p.error('reference mismatch: '+key)
    def identity(r):return {k:v for k,v in (line.split('=',1) for line in r['conditioning_identity'].splitlines()) if k!='arithmetic-environment' and not k.startswith('shader-override')}
    if identity(old)!=identity(records['B6-off']):p.error('reference model/conditioning identity mismatch')
    payloads=conditioning_payloads(records['B6-off'])
    if conditioning_payloads(old)!=payloads:p.error('original reference has different conditioning tensors')
    if sha(reference/'result.mp4')!=old['media_sha256']:p.error('changed original reference')
    controls=json.loads(a.controls.read_text())
    if not controls.get('calibration_pass') or controls['contract_sha256']!=sha(ROOT/'tests/metal_reference_contract.json') or controls['reference_sha256']!=old['media_sha256'] or controls['metric_script_sha256']!=sha(ROOT/'scripts/metal_reference_quality.py'):p.error('invalid or stale quality controls')
    out.mkdir(parents=True)
    report={'schema':1,'geometry':[640,480,243],'binary_sha256':index['binary_sha256'],
        'index_sha256':sha(runs/'index.json'),'script_sha256':sha(__file__),
        'reference_record_sha256':sha(reference/'record.json'),'contract_sha256':sha(ROOT/'tests/metal_reference_contract.json'),
        'reference_conditioning_payload_sha256':payloads,
        'calibration_controls_sha256':sha(a.controls),
        'modes':{},'quality':{},'unrun':index['unrun'],'hardware_trace':{},'human_review':'not a gate; no human observations claimed'}
    baseline=records['B5-off']['timing']
    for mode in MODES:
        r=records['B5-'+mode];timing=r['timing'];ane=r['ane']
        # The frozen prototype's probe-only log included incoming command
        # draining in region_seconds and named QKV wall time gpu_seconds.
        # Normalize those records; current logging separates the intervals.
        normalized=[]
        for row in ane:
            row=dict(row)
            if row.get('probe') and 'gpu_wait_seconds' not in row:
                row['gpu_wait_seconds']=row['gpu_seconds']
                row['drain_seconds']=max(0,row['region_seconds']-row['gpu_seconds'])
                row['region_seconds']=row['gpu_seconds']
            normalized.append(row)
        m={'timing':timing,'incremental_b5_speedup':baseline['steady_median']/timing['steady_median'],
            'conservative_b5_speedup':baseline['steady_minimum']/timing['steady_maximum'],
            'steady_ranges_disjoint':timing['steady_maximum']<baseline['steady_minimum'],
            'peak_footprint_gib':max(s['physical_footprint_bytes'] for s in r['steps'])/2**30,
            'footprint_sampling':'maximum of end-of-step samples; transient driver allocations may be higher',
            'peak_sampled_rss_gib':r['peak_sampled_resident_bytes']/2**30,
            'peak_metal_gib':max(s['metal_peak_bytes'] for s in r['steps'])/2**30,
            'peak_system_wired_gib':max((s.get('system_wired_bytes',0) for s in ane),default=0)/2**30 or None,
            'ane_setup':r['ane_setup'],'offloaded_blocks':r.get('ane_offloaded_blocks',0),
            'setup_by_case':{case:records[case+'-'+mode]['ane_setup'] for case in ('B1','B5','B6')},
            'recovered_blocks':r.get('ane_recovered_blocks',0),
            'ane_hardware_ready':r.get('ane_hardware_ready',False),
            'frozen_rows':{b:sorted({s['ane_rows'] for s in ane if s['block']==b}) for b in range(50)},
            'ane_region_totals':{key:sum(s.get(key,0) for s in normalized) for key in ('pack_seconds','predict_seconds','unpack_seconds','drain_seconds','region_seconds','gpu_wait_seconds','join_seconds','host_predict_overlap_seconds')}}
        m['speed_gate_pass']=mode!='off' and m['conservative_b5_speedup']>=1.10 and m['steady_ranges_disjoint']
        saving=baseline['steady_median']-timing['steady_median']
        setup=sum(s['load_seconds'] for s in r['ane_setup'])
        m['setup_amortization_steps']=setup/saving if saving>0 and m['steady_ranges_disjoint'] else None
        m['serial_static_same_av']={case:records[case+'-serial']['av_sha256']==records[case+'-static']['av_sha256'] for case in ('B1','B5','B6')}
        report['modes'][mode]=m
    # Retain both incremental quality and the original Reference+ comparison.
    # A prior acceptance of SOL does not make a new ANE precision change pass.
    for anchor,ref,modes in [('incremental',runs/'B6-off',MODES[1:]),('reference-plus',reference,MODES)]:
        for mode in modes:
            name=anchor+'-'+mode;target=out/name
            cmd=[sys.executable,str(ROOT/'scripts/metal_reference_quality.py'),str(ref/'result.mp4'),
                '--candidate',str(runs/f'B6-{mode}'/'result.mp4'),'--output',str(target)]
            with (out/(name+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode not in (0,3) or not (target/'metrics.json').exists():raise ValueError('quality tool failure: '+name)
            metrics=json.loads((target/'metrics.json').read_text())
            if metrics['reference_sha256']!=sha(ref/'result.mp4') or metrics['candidate_sha256']!=records['B6-'+mode]['media_sha256']:raise ValueError('quality input mismatch')
            report['quality'][name]={'screen_pass':metrics['screen_pass'],'checks':metrics['results']['candidate']['checks'],
                'metrics_sha256':sha(target/'metrics.json'),'path':str(target/'review.html')}
    # Numerical drift is diagnostic; the frozen Reference perceptual contract
    # remains the promotion gate. Capture every step, including audio/latents.
    for case,folder in [('B1','boundaries'),('B6','steps')]:
        for mode in MODES[1:]:
            name=f'{case}-{mode}-tensors'
            cmd=[sys.executable,str(ROOT/'scripts/metal_native_tensor_compare.py'),str(runs/f'{case}-off'/folder),
                 str(runs/f'{case}-{mode}'/folder),'--output',str(out/(name+'.json'))]
            with (out/(name+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if not (out/(name+'.json')).exists():raise ValueError('tensor tool failure')
    trace=runs/'B1-static/metal.trace';overlap=runs/'B1-static/trace-overlap.json'
    report['hardware_trace']={'path':str(trace),'recorded':records['B1-static'].get('trace_pass',False),
        'overlap':json.loads(overlap.read_text()) if overlap.exists() else None}
    evidence=report['hardware_trace']['overlap']
    if evidence:
        for name,digest in evidence['input_sha256'].items():
            if sha(overlap.parent/name)!=digest:raise ValueError('changed trace export')
        report['hardware_trace']['bundle_sha256']={str(p.relative_to(trace)):sha(p) for p in sorted(trace.rglob('*')) if p.is_file()}
    for mode in MODES[1:]:
        report['modes'][mode]['promotion_pass']=(report['modes'][mode]['speed_gate_pass'] and
            report['quality']['incremental-'+mode]['screen_pass'] and report['quality']['reference-plus-'+mode]['screen_pass'] and
            report['modes'][mode]['ane_hardware_ready'] and report['modes'][mode]['offloaded_blocks']>0 and
            bool(evidence and evidence['hardware_overlap_observed']))
    report['disposition']='Retain GPU-only default; ANE remains opt-in.'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def link(path):return html.escape(os.path.relpath(path,out),quote=True)
    body=['<!doctype html><meta charset="utf-8"><title>M3C GPU/ANE — 243 frames</title>',
        '<style>body{font:16px system-ui;background:#171920;color:#eee;max-width:1280px;margin:2rem auto;padding:0 1rem}.pair{display:grid;grid-template-columns:1fr 1fr;gap:1rem}video{width:100%}a{color:#9cd3ff}td,th{padding:.5rem;text-align:right}pre{white-space:pre-wrap}button{padding:.5rem}</style>',
        '<h1>M3C GPU/ANE QKV comparison</h1><p>640×480 · 243 frames · all 50 blocks · BF16 source weights/state · identical conservative SOL recipe. B5 timing: four steady evaluations. B6 playback: six evaluations and original VAE.</p>',
        '<p>'+html.escape(report['disposition'])+' Automated screens are distinct from human observations. No 50-step, conditioning/continuation corpus or M5 qualification is claimed.</p>',
        '<table><tr><th>Mode</th><th>B5 seconds/step</th><th>Incremental gain</th><th>Max step footprint GiB</th><th>1.10× speed gate</th><th>Incremental quality screen</th><th>Reference+ screen</th></tr>']
    for mode,m in report['modes'].items():
        incremental=report['quality']['incremental-'+mode]['screen_pass'] if mode!='off' else 'baseline'
        original=report['quality']['reference-plus-'+mode]['screen_pass']
        body.append(f'<tr><td>{mode}</td><td>{m["timing"]["steady_median"]:.3f}</td><td>{m["incremental_b5_speedup"]:.4f}×</td><td>{m["peak_footprint_gib"]:.3f}</td><td>{m["speed_gate_pass"]}</td><td>{incremental}</td><td>{original}</td></tr>')
    body.append('</table>')
    for mode in MODES[1:]:
        body+=['<h2>GPU-only vs '+mode+'</h2><div class="pair">']
        for name in ('off',mode):body.append(f'<section><h3>{name}</h3><video controls preload="metadata" {"muted" if name!="off" else ""} src="{link(runs/f"B6-{name}"/"result.mp4")}"></video></section>')
        body+=['</div><button onclick="play(this)">Play pair</button> <button onclick="pause(this)">Pause pair</button> <button onclick="audio(this,0)">GPU-only audio</button> <button onclick="audio(this,1)">ANE candidate audio</button>',
            f'<p><a href="incremental-{mode}/review.html">Incremental quality measurements</a> · <a href="reference-plus-{mode}/review.html">Original MPSGraph Reference+ comparison</a></p>']
    body+=['<p><a href="reference-plus-off/review.html">GPU-only SOL vs original MPSGraph Reference+</a> · <a href="report.json">Verified records and gates</a></p>',
        '<details><summary>Evidence, gates and hashes</summary><pre>'+html.escape(json.dumps(report,indent=2))+'</pre></details>',
        '<script>function videos(b){let e=b.previousElementSibling;while(!e.classList.contains("pair"))e=e.previousElementSibling;return [...e.querySelectorAll("video")]}function play(b){let v=videos(b);v[1].currentTime=v[0].currentTime;v.forEach(x=>x.play())}function pause(b){videos(b).forEach(x=>x.pause())}function audio(b,which){videos(b).forEach((v,i)=>v.muted=i!==which)}</script>']
    (out/'review.html').write_text('\n'.join(body)+'\n');print(out/'review.html')

if __name__=='__main__':main()
