#!/usr/bin/env python3
"""Verify M6 records, isolate weight/routing gains, and create playback pages."""
import argparse
from array import array
import html
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
from metal_native_bench import ROOT, CASES, sha
from metal_q8_validate import MODES

SCREEN_FILES=('tests/metal_reference_contract.json','scripts/metal_reference_quality.py',
              'tests/requirements-metal-quality.txt')

def verify_screens(source_hashes,root=ROOT):
    for name in SCREEN_FILES:
        if name not in source_hashes or sha(root/name)!=source_hashes[name]:
            raise ValueError('quality contract changed after frozen snapshot: '+name)

def metric(a,b):
    if len(a)!=len(b) or not a:raise ValueError('tensor shape mismatch')
    ss=aa=dot=error=0.; maximum=0.
    for x,y in zip(a,b):
        if not math.isfinite(x) or not math.isfinite(y):raise ValueError('nonfinite tensor')
        ss+=x*x;aa+=y*y;dot+=x*y;error+=(x-y)**2;maximum=max(maximum,abs(x-y))
    return {'relative_l2':math.sqrt(error/max(ss,1e-30)),
            'cosine':dot/math.sqrt(ss*aa) if ss*aa else float(ss==aa),
            'max_abs':maximum}

def floats(data):
    f=array('f');f.frombytes(data)
    if sys.byteorder!='little':f.byteswap()
    return f

def av(path):
    data=path.read_bytes()
    if data[:8]!=b'H3AV\r\n\x1a\n' or len(data)<160:raise ValueError('invalid AV state')
    nv,na=struct.unpack_from('<QQ',data,72)
    if len(data)!=160+nv+na or nv%4 or na%4:raise ValueError('invalid AV shape')
    return (floats(data[160:160+nv]),floats(data[160+nv:]))

def validate_tensors(run, record):
    path=run/'result.h3av'
    with path.open('rb') as stream:header=stream.read(160)
    if list(struct.unpack_from('<3I',header,24))!=record['geometry']:
        raise ValueError('AV geometry mismatch')
    record['av_identity']=header[24:128].hex()
    final=av(path)
    if any(not math.isfinite(v) for values in final for v in values):
        raise ValueError('nonfinite final state')
    if record['case'] not in ('B1','B6'):return
    expected={f'step-{s:03d}-{m}-{kind}.f32':len(final[i])*4
              for s in range(1,record['evaluations']+1) for i,m in enumerate(('video','audio'))
              for kind in ('input','velocity','latent')}
    if set(record.get('step_tensor_sha256',{}))!=set(expected):
        raise ValueError('incomplete step diagnostics')
    for name,length in expected.items():
        data=(run/'steps'/name).read_bytes()
        if len(data)!=length or any(not math.isfinite(v) for v in floats(data)):
            raise ValueError('invalid diagnostic tensor '+name)

def validate_record(run, entry, binary_sha):
    for name,digest in entry['sha256'].items():
        if sha(run/name)!=digest:raise ValueError('changed artifact '+str(run/name))
    r=json.loads((run/'record.json').read_text())
    if (not r['evaluation_contract_pass'] or r['returncode'] or r['timed_out'] or
        r['geometry']!=[640,480,243] or r['blocks']!=50 or r['evaluations']!=CASES[r['case']] or
        r['sampler']!='cpu-euler' or r.get('teacher_from') or
        r['manifest']['binary_sha256']!=binary_sha or sha(run/'bin/h3cli')!=binary_sha):
        raise ValueError('invalid M6 workload '+str(run))
    if r['av_sha256']!=sha(run/'result.h3av') or r['media_sha256']!=sha(run/'result.mp4'):
        raise ValueError('stale result identity')
    if r['case']=='B6' and (r['preview_vae'] or r['evaluations']!=6):raise ValueError('invalid B6 decode/schedule')
    if r['case']=='B5' and any(r.get(k) for k in ('capture_steps','component_fences','region_fences','trace_command','capture_ranges')):
        raise ValueError('diagnostic B5 timing')
    if len(r['steps'])!=r['evaluations'] or [x['step'] for x in r['steps']]!=list(range(1,r['evaluations']+1)):
        raise ValueError('missing/repeated step')
    if any(not x['evaluated'] or not math.isfinite(x['wall_seconds']) or x['wall_seconds']<=0 for x in r['steps']):
        raise ValueError('invalid timing/evaluation')
    if r['weight_precision']=='q8' and not r.get('q8_execution_pass'):raise ValueError('missing Q8 dispatch proof')
    for name,digest in r.get('step_tensor_sha256',{}).items():
        if sha(run/'steps'/name)!=digest:raise ValueError('changed diagnostic tensor')
    validate_tensors(run,r)
    command=r['command'];flag='--load-conditioning' if '--load-conditioning' in command else '--save-conditioning'
    if sha(command[command.index(flag)+1])!=r['conditioning_sha256_after']:raise ValueError('changed conditioning')
    memory=[json.loads(x) for x in re.findall(r'h3_memory (\{[^\n]+\})',(run/'run.log').read_text())]
    r['peak_footprint_bytes']=max((x['footprint_bytes'] for x in memory),default=0)
    if r['peak_footprint_bytes']<=0:raise ValueError('missing memory profile '+str(run))
    r['memory_measurement']='maximum observed physical footprint at instrumented boundaries'
    r['step_footprint_bytes']=[max(x['footprint_bytes'] for x in memory if x['step']==s)
                               for s in range(r['evaluations'])]
    r['phase_seconds']={}
    for phase,seconds in re.findall(r'h3(?:cli)?: phase duration ([^\n]+?): ([\d.]+) s',(run/'run.log').read_text()):
        r['phase_seconds'][phase]=r['phase_seconds'].get(phase,0)+float(seconds)
    return r

def matched(a,b,quant=False):
    for key in ('case','evaluations','blocks','geometry','seed','prompt','sampler','preview_vae',
                'reference_sha256','extra_inputs_sha256','conditioning_identity','conditioning_sha256_after','av_identity'):
        if a.get(key)!=b.get(key):raise ValueError('workload mismatch '+key)
    def arithmetic(record):
        return {k:v for k,v in record['manifest']['environment'].items()
                if not k.startswith(('H3_TEST_','H3_DEBUG_','H3_PROFILE'))}
    if arithmetic(a)!=arithmetic(b):raise ValueError('arithmetic environment mismatch')
    if a['case'] in ('B1','B6'):
        for modality in ('video','audio'):
            key=f'step-001-{modality}-input.f32'
            if a['step_tensor_sha256'][key]!=b['step_tensor_sha256'][key]:
                raise ValueError('initial noise/conditioning state mismatch')
    if quant:
        if (a['weight_precision'],b['weight_precision'])!=('bf16','q8'):raise ValueError('expected BF16/Q8 ablation')
        x=dict(a['metal_options']);y=dict(b['metal_options'])
        x.pop('metal-weight-format');y.pop('metal-weight-format')
        if x!=y or a['attention']!=b['attention']:raise ValueError('attention changed in Q8 ablation')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--partial',action='store_true',help='Publish available evidence without promotion')
    p.add_argument('--quality',action='store_true',help='Run frozen B6 perceptual metrics and reference controls')
    a=p.parse_args();runs=a.runs.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    index=json.loads((runs/'index.json').read_text());limits_path=ROOT/'tests/metal_q8_limits.json'
    limits=json.loads(limits_path.read_text())
    if index['contract_sha256']!=sha(limits_path):p.error('changed Q8 contract')
    names={case+'-'+mode for case in ('B1','B5','B6') for mode in MODES}
    if not a.partial and set(index['runs'])!=names:p.error('incomplete suite; use --partial to inspect')
    records={name:validate_record(runs/name,entry,index['binary_sha256']) for name,entry in index['runs'].items()}
    snapshots=set()
    for r in records.values():
        frozen=r['manifest']['binary_build_provenance']
        if not frozen:raise ValueError('missing build provenance')
        if frozen['source_snapshot'] not in snapshots:
            verify_screens(frozen['source_sha256'])
            for name,digest in frozen['source_sha256'].items():
                if sha(Path(frozen['source_snapshot'])/name)!=digest:raise ValueError('changed source snapshot')
            snapshots.add(frozen['source_snapshot'])
    report={'schema':1,'geometry':[640,480,243],'binary_sha256':index['binary_sha256'],
            'index_sha256':sha(runs/'index.json'),'limits_sha256':sha(limits_path),'script_sha256':sha(__file__),
            'missing_runs':sorted(names-set(records)),'ablations':{},'quality':{},
            'measured_subset_pass':False,'production_qualified':False,
            'hardware_ratios':None,'hardware_comparison_status':'skipped at user request; no CUDA comparison or hardware ratio claimed',
            'hardware_tiers':{'reference-plus':'measured locally','reference-dense-fp16':'measured locally',
                              'reference-sol':'measured locally','q8-dense':'measured locally','q8-sol':'measured locally',
                              'ane':'prior prototype failed promotion; Q8 + ANE unsupported',
                              'native-dense-bf16':'optional deferred branch','preview-turbo':'outside this original-checkpoint M6 ablation'},
            'measurement_notes':['One B5 trajectory per mode; small timing differences do not establish a repeatable speedup.',
                                 'A CPU-only final source build overlapped part of B5-sol-q8; GPU workloads were serial. Retain the raw timings and conservative comparison.'],
            'excluded':['Q8 + ANE (unsupported)','M5 arithmetic (unqualified)','50-step trajectories','held-out conditioning corpus']}
    pairs=[('q8-dense','dense-bf16','dense-q8'),('q8-sol','sol-bf16','sol-q8'),
           ('attention','reference-plus','dense-bf16'),('sol','dense-bf16','sol-bf16'),
           ('total-q8-dense','reference-plus','dense-q8'),('total-q8-sol','reference-plus','sol-q8')]
    labels={'reference-plus':'Reference+ (MPSGraph BF16)',
            'dense-bf16':'Dense FP16 attention / BF16 weights','dense-q8':'Dense FP16 attention / Q8 weights',
            'sol-bf16':'SOL FP16 attention / BF16 weights','sol-q8':'SOL FP16 attention / Q8 weights'}
    for label,baseline,candidate in pairs:
        result={}
        for case in ('B1','B5','B6'):
            left=case+'-'+baseline;right=case+'-'+candidate
            if left not in records or right not in records:continue
            r=records[left];c=records[right];matched(r,c,label.startswith('q8-'))
            left_av=av(runs/left/'result.h3av');right_av=av(runs/right/'result.h3av')
            row={'reference':left,'candidate':right,'final_latents':{kind:metric(x,y) for kind,x,y in zip(('video','audio'),left_av,right_av)},
                 'peak_footprint_ratio':c['peak_footprint_bytes']/r['peak_footprint_bytes'],
                 'reference_peak_bytes':r['peak_footprint_bytes'],'candidate_peak_bytes':c['peak_footprint_bytes'],
                 'memory_measurement':c['memory_measurement'],
                 'reference_step_footprint_bytes':r['step_footprint_bytes'],'candidate_step_footprint_bytes':c['step_footprint_bytes'],
                 'reference_timing':r['timing'],'candidate_timing':c['timing'],
                 'reference_phases':r['phase_seconds'],'candidate_phases':c['phase_seconds'],
                 'timing_scope':'B5 ordinary steps 2..5; no component/range diagnostics' if case=='B5' else 'diagnostic capture; use B5 for performance claims'}
            row['execution']={}
            for name,rec in (('reference',r),('candidate',c)):
                row['execution'][name]={'attention':rec['attention'],'weight_precision':rec['weight_precision'],
                    'composition':rec['execution_composition'],'metal_options':rec['metal_options'],
                    'invalid_heads':sum(x['invalid_heads'] for x in rec.get('mixed',[])),
                    'recovered_heads':sum(x['recovered_heads'] for x in rec.get('mixed',[])),
                    'sol_exact_pairs':sum(x['exact'] for x in rec.get('sol',[])),
                    'sol_approximate_pairs':sum(x['approximate'] for x in rec.get('sol',[])),
                    'record':str(runs/(left if name=='reference' else right)/'record.json')}
            if case=='B1' and label.startswith('q8-'):
                row['velocities']={kind:metric(floats((runs/left/'steps'/f'step-001-{kind}-velocity.f32').read_bytes()),
                                               floats((runs/right/'steps'/f'step-001-{kind}-velocity.f32').read_bytes())) for kind in ('video','audio')}
                row['velocity_screen_pass']=all(x['relative_l2']<=limits['B1_velocity_relative_l2_max'] for x in row['velocities'].values())
            if case=='B5':
                ref=[s['wall_seconds'] for s in r['steps'][1:]];cand=[s['wall_seconds'] for s in c['steps'][1:]]
                row.update(steady_speedup=statistics.median(ref)/statistics.median(cand),
                           conservative_speedup=min(ref)/max(cand),
                           time_screen_pass=statistics.median(cand)/statistics.median(ref)<=limits['B5_maximum_step_time_ratio'])
            if label.startswith('q8-'):
                w=c['weights'][-1];row['weights']=w
                row['storage_screen_pass']=w['storage_bytes']/w['source_bytes']<=limits['weight_storage_ratio_max']
            result[case]=row
        report['ablations'][label]=result
    if a.quality:
        if 'B6-reference-plus' not in records:p.error('quality requires completed Reference+ B6')
        def quality(name,reference,candidate=None):
            target=out/name;metrics_path=target/'metrics.json'
            if not metrics_path.exists():
                command=[sys.executable,str(ROOT/'scripts/metal_reference_quality.py'),str(reference),
                         '--output',str(target)]+(['--candidate',str(candidate)] if candidate else ['--calibrate'])
                with (out/(name+'.log')).open('w') as log:status=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
                if status not in (0,3) or not metrics_path.exists():raise ValueError('quality runner failed '+name)
            m=json.loads(metrics_path.read_text())
            if (m['reference_sha256']!=sha(reference) or m.get('candidate_sha256')!=(sha(candidate) if candidate else None) or
                m['contract_sha256']!=sha(ROOT/'tests/metal_reference_contract.json') or
                m['metric_script_sha256']!=sha(ROOT/'scripts/metal_reference_quality.py')):raise ValueError('stale quality result')
            key='screen_pass' if candidate else 'calibration_pass'
            if key not in m:raise ValueError('incomplete quality result')
            return {'pass':m[key],'metrics_sha256':sha(metrics_path),'page':str(target/'review.html')}
        report['quality']['controls']=quality('controls',runs/'B6-reference-plus/result.mp4')
        for label,left,right in pairs:
            if label in ('attention','sol'):continue
            if 'B6-'+left in records and 'B6-'+right in records:
                report['quality'][label]=quality('quality-'+label,runs/('B6-'+left)/'result.mp4',runs/('B6-'+right)/'result.mp4')
        report['measured_subset_pass']=not report['missing_runs'] and report['quality']['controls']['pass'] and all(
            report['ablations'][label]['B1']['velocity_screen_pass'] and report['ablations'][label]['B5']['time_screen_pass'] and
            report['ablations'][label]['B5']['storage_screen_pass'] and report['quality'][label]['pass'] and
            report['quality']['total-'+label]['pass'] for label in ('q8-dense','q8-sol'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def video(path,muted=False):
        return '<video controls preload="metadata" '+('muted ' if muted else '')+'src="'+html.escape(os.path.relpath(path,out),quote=True)+'"></video>'
    style='<style>body{font:16px system-ui;background:#181a20;color:#eee;margin:2rem;line-height:1.5}a{color:#9cf}.pair{display:grid;grid-template-columns:1fr 1fr;gap:1rem}video{width:100%}pre{white-space:pre-wrap;font-size:12px}th,td{text-align:left;padding:.5rem 1rem;border-bottom:1px solid #555}table{border-collapse:collapse}button{font:inherit;margin:.5rem;padding:.3rem .8rem}@media(max-width:700px){.pair{grid-template-columns:1fr}body{margin:1rem}}</style>'
    controls='''<p><button onclick="startPair()">Play both from start</button><button onclick="pausePair()">Pause both</button>
<button onclick="audioSide(0)">Reference audio</button><button onclick="audioSide(1)">Candidate audio</button></p>
<script>
const players=[...document.querySelectorAll('video')];
function startPair(){players.forEach(v=>{v.currentTime=0;v.play().catch(()=>{});});}
function pausePair(){players.forEach(v=>v.pause());}
function audioSide(index){players.forEach((v,i)=>v.muted=i!==index);}
</script>'''
    cards=[]
    for label,left,right in pairs:
        if 'B6-'+left in records and 'B6-'+right in records:
            page=label+'.html'
            (out/page).write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+style+'<h1>M6 '+html.escape(label)+'</h1><p>243 frames / 6 evaluations / production VAE. Left: '+html.escape(labels[left])+'; right: '+html.escape(labels[right])+'. Reference audio is selected initially.</p><div class="pair">'+video(runs/('B6-'+left)/'result.mp4')+video(runs/('B6-'+right)/'result.mp4',True)+'</div>'+controls+'<p><a href="review.html">All comparisons</a></p>')
            cards.append('<li><a href="'+page+'">'+html.escape(label)+'</a></li>')
    table=['<table><thead><tr><th>Mode</th><th>B5 median step (s)</th><th>Observed footprint (GB)</th><th>Core load (s)</th></tr></thead><tbody>']
    for mode in MODES:
        r=records.get('B5-'+mode)
        if r:table.append('<tr><td><a href="'+html.escape(os.path.relpath(runs/('B5-'+mode)/'record.json',out),quote=True)+'">'+html.escape(labels[mode])+'</a></td><td>'+format(r['timing']['steady_median'],'.2f')+'</td><td>'+format(r['peak_footprint_bytes']/1e9,'.2f')+'</td><td>'+format(r['phase_seconds'].get('load transformer core',0),'.2f')+'</td></tr>')
    table.append('</tbody></table>')
    quality_links=[]
    for label,q in report['quality'].items():
        quality_links.append('<li><a href="'+html.escape(os.path.relpath(q['page'],out),quote=True)+'">'+html.escape(label)+'</a>: '+('PASS' if q['pass'] else 'FAILED frozen screen')+'</li>')
    status=('Validation in progress: '+str(len(report['missing_runs']))+' runs pending.' if report['missing_runs'] else
            'Quality screens have not run.' if not report['quality'] else
            'The measured subset passed the frozen screens.' if report['measured_subset_pass'] else
            'The measured subset failed one or more frozen screens; Q8 remains opt-in.')
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+style+'<h1>M6 Q8 results</h1><p><strong>'+status+'</strong></p><p>Unqualified; no human review gate. Storage, attention and routing gains are reported separately. Six evaluations do not establish 50-step equivalence.</p><p>640×480, 243 frames, 50 blocks. GB uses decimal units. Memory is the maximum observed physical footprint at instrumented boundaries. CUDA comparison was skipped at the user’s request.</p>'+''.join(table)+'<h2>Paired playback</h2><ul>'+''.join(cards)+'</ul><h2>Automated quality screens</h2><ul>'+''.join(quality_links)+'</ul><p><a href="report.json">Verified records and gates</a></p><details><summary>Detailed results</summary><pre>'+html.escape(json.dumps(report,indent=2))+'</pre></details>')
    print(json.dumps({'measured_subset_pass':report['measured_subset_pass'],
                     'production_qualified':report['production_qualified'],
                     'missing_runs':report['missing_runs'],'gallery':str(out/'review.html')}))

if __name__=='__main__':main()
