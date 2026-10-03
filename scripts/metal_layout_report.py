#!/usr/bin/env python3
"""Verify retained M2E evidence and produce final local playback/report artifacts."""
import argparse
import html
import json
from pathlib import Path
import re
import statistics
from metal_native_bench import sha

def load_run(path,expected_hash):
    if sha(path/'record.json')!=expected_hash:raise ValueError('changed record: '+str(path))
    r=json.loads((path/'record.json').read_text())
    if not r['evaluation_contract_pass'] or r['geometry']!=[640,480,243] or r['blocks']!=50:raise ValueError('invalid workload')
    for name,key in [('bin/h3cli','binary_sha256')]:
        if sha(path/name)!=r['manifest'][key]:raise ValueError('changed binary')
    for name,key in [('result.h3av','av_sha256'),('result.mp4','media_sha256')]:
        if sha(path/name)!=r[key]:raise ValueError('changed output')
    for name,digest in r.get('step_tensor_sha256',{}).items():
        if sha(path/'steps'/name)!=digest:raise ValueError('changed captured sampler tensor')
    frozen=r['manifest']['binary_build_provenance']
    if not frozen or frozen['binary_sha256']!=r['manifest']['binary_sha256']:raise ValueError('missing executable provenance')
    for name,digest in frozen['source_sha256'].items():
        if sha(Path(frozen['source_snapshot'])/name)!=digest:raise ValueError('changed frozen source')
    if r['metal_options']['metal-attention-layout']=='fused' and not r.get('layout_pass'):raise ValueError('missing fusion evidence')
    if any(row['invalid_heads'] for row in r['mixed']):raise ValueError('invalid attention heads')
    return r

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--accepted-b6',type=Path,required=True)
    a=p.parse_args();runs=a.runs.resolve();out=a.output.resolve()
    if out.exists():p.error('use a fresh report directory')
    index=json.loads((runs/'index.json').read_text())
    if len(index['pairs'])!=4:p.error('incomplete suite')
    if sha(runs/'operators.jsonl')!=index['operators_sha256']:p.error('changed operator records')
    operators=[json.loads(line) for line in (runs/'operators.jsonl').read_text().splitlines()]
    if len(operators)!=12 or not all(r['pass'] and r['byte_identical'] for r in operators):p.error('operator failure')
    if {(x['attention'],x['case']) for x in index['pairs']}!={('dense','B1'),('dense','B5'),('dense','B6'),('sol','B5')}:p.error('incomplete suite')
    report={'schema':1,'geometry':[640,480,243],'index_sha256':sha(runs/'index.json'),'script_sha256':sha(__file__),
            'arithmetic_recipe':4,'layout_recipe':1,'checkpoint_version':5,'pairs':[],
            'unrun':['held-out conditioning/continuation renders','SOL B6','M5','ANE'],
            'default':'MPSGraph Reference+; native adapter remains default layout'}
    all_runs={}
    for pair in index['pairs']:
        kind,case=pair['attention'],pair['case'];records={layout:load_run(runs/f'{kind}-{case}-{layout}',digest) for layout,digest in pair['record_sha256'].items()}
        left,right=records['adapter'],records['fused'];all_runs[(kind,case)]=records
        if any(r['manifest']['binary_sha256']!=index['binary_sha256'] for r in records.values()):raise ValueError('unmatched suite executable')
        if sha(Path(left['original_binary']).parent/'bin/metal_layout')!=index['operator_binary_sha256']:raise ValueError('changed operator binary')
        if left['sol']!=right['sol'] or left['sol_policy']!=right['sol_policy']:raise ValueError('fusion changed SOL decisions or protection counters')
        if left['mixed']!=right['mixed']:raise ValueError('fusion changed per-block range/recovery records')
        if left['av_sha256']!=right['av_sha256']:raise ValueError('fusion changed AV state')
        if left.get('step_tensor_sha256')!=right.get('step_tensor_sha256'):raise ValueError('fusion changed sampler tensors')
        r={**pair,'execution':{},'layout':{},'incremental_speed_claim':False,'range_records_identical':True,'routing_records_identical':True}
        for layout,record in records.items():
            r['execution'][layout]={k:statistics.median(x[k] for x in record['execution'][1:] or record['execution']) for k in ('dispatches','blit_copies','explicit_d2d_bytes','encode_seconds','wait_seconds')}
            r['layout'][layout]=record['layout']
        r['peak_saving_gib']=pair['peak_gib']['adapter']-pair['peak_gib']['fused']
        if case=='B5':
            r['paired_steady_speedups']=[x['wall_seconds']/y['wall_seconds'] for x,y in zip(left['steps'][1:],right['steps'][1:])]
            # Descriptive separation only, not a confidence interval or a new
            # replacement for the previously frozen whole-model speed gate.
            r['steady_ranges_disjoint']=left['timing']['steady_minimum']>right['timing']['steady_maximum']
            r['incremental_speed_claim']=r['steady_ranges_disjoint'] and min(r['paired_steady_speedups'])>1
        if case=='B1':
            r['boundary_sha256']={layout:{p.name:sha(p) for p in (runs/f'{kind}-{case}-{layout}'/'boundaries').glob('*.bf16')} for layout in records}
            if not r['boundary_sha256']['adapter'] or r['boundary_sha256']['adapter']!=r['boundary_sha256']['fused']:raise ValueError('changed block boundaries')
            r['regions']={}
            for layout in records:
                rows=records[layout]['components']
                r['regions'][layout]={name:statistics.median(x['wall_seconds'] for x in rows if x['name']==name) for name in {x['name'] for x in rows}}
        report['pairs'].append(r)
    accepted=a.accepted_b6.resolve();old=json.loads((accepted/'record.json').read_text())
    if sha(accepted/'result.h3av')!=old['av_sha256']:raise ValueError('changed accepted state')
    if sha(accepted/'result.mp4')!=old['media_sha256']:raise ValueError('changed accepted media')
    new=all_runs[('dense','B6')]['fused']
    report['accepted_dense_b6']={'path':str(accepted),'record_sha256':sha(accepted/'record.json'),
        'same_final_av_state':old['av_sha256']==new['av_sha256'],
        'same_production_vae_mp4':old['media_sha256']==new['media_sha256'],
        'same_sampler_tensors':old.get('step_tensor_sha256')==new.get('step_tensor_sha256')}
    if not all(report['accepted_dense_b6'][k] for k in ('same_final_av_state','same_sampler_tensors','same_production_vae_mp4')):
        raise ValueError('retained accepted dense B6 arithmetic changed; inspect before claiming inherited acceptance')
    dense=all_runs[('dense','B5')];sol=all_runs[('sol','B5')]
    report['cumulative_b5_speedup_over_dense_adapter']=dense['adapter']['timing']['steady_median']/sol['fused']['timing']['steady_median']
    report['sol_b5_speedup_over_dense_adapter']=dense['adapter']['timing']['steady_median']/sol['adapter']['timing']['steady_median']
    routes=sol['fused']['sol'];report['sol_actual_exact_fraction']=sum(x['exact'] for x in routes)/sum(x['exact']+x['approximate'] for x in routes)
    out.mkdir(parents=True);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    import os
    def link(path):return html.escape(os.path.relpath(path,out),quote=True)
    rows=['<!doctype html><meta charset="utf-8"><title>M2E layout fusion — 243 frames</title>',
        '<style>body{font:16px system-ui;background:#161920;color:#eee;max-width:1250px;margin:2rem auto;padding:0 1rem}video{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:1rem}a{color:#9cd3ff}td,th{padding:.5rem;text-align:right}pre{white-space:pre-wrap}button{padding:.5rem}</style>',
        '<h1>M2E layout fusion</h1><p>640×480 · 243 frames · all 50 blocks · original BF16 weights/state. Dense B6: six evaluations and production VAE. SOL B5: five evaluations and preview VAE, for timing and arithmetic comparison.</p>',
        '<p>The fused and adapter paths produce byte-identical final AV states. Dense B6 also matches the previously accepted six-evaluation result. Earlier quality/performance screen failures and deferred conditioning cases remain recorded; this page does not claim 50-step or M5 qualification.</p>',
        '<table><tr><th>Case</th><th>Adapter steady seconds</th><th>Fused steady seconds</th><th>Speedup</th><th>Peak saving GiB</th><th>Separated timing ranges</th></tr>']
    for r in report['pairs']:
        if r['case']!='B5':continue
        rows.append(f'<tr><td>{r["attention"]} B5</td><td>{r["timing"]["adapter"]["steady_median"]:.3f}</td><td>{r["timing"]["fused"]["steady_median"]:.3f}</td><td>{r["steady_speedup"]:.4f}×</td><td>{r["peak_saving_gib"]:.3f}</td><td>{r["steady_ranges_disjoint"]}</td></tr>')
    rows+=['</table>']
    for kind,case in [('dense','B6'),('sol','B5')]:
        rows+=['<h2>'+kind.upper()+' '+case+'</h2><div class="pair">']
        for layout in ('adapter','fused'):
            rows.append(f'<section><h3>{layout.capitalize()}</h3><video controls preload="metadata" {"muted" if layout=="fused" else ""} src="{link(runs/f"{kind}-{case}-{layout}"/"result.mp4")}"></video></section>')
        rows+=['</div><button onclick="playPair(this)">Play pair</button> <button onclick="pausePair(this)">Pause pair</button>']
    rows+=['<p><a href="report.json">Verified measurements and evidence hashes</a></p><pre>'+html.escape(json.dumps(report,indent=2))+'</pre>',
        '<script>function pair(b){let e=b.previousElementSibling;while(!e.classList.contains("pair"))e=e.previousElementSibling;return [...e.querySelectorAll("video")]}function playPair(b){let v=pair(b);v[1].currentTime=v[0].currentTime;v.forEach(x=>x.play())}function pausePair(b){pair(b).forEach(x=>x.pause())}</script>']
    (out/'review.html').write_text('\n'.join(rows)+'\n');print(out/'review.html')
if __name__=='__main__':main()
