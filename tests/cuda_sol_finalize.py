#!/usr/bin/env python3
"""GPU-free closing audit. Preserve original records and emit derived evidence."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import json,re,subprocess,time,os
from fractions import Fraction
from pathlib import Path
import psutil
from cuda_sol_campaign import ROOT,ACCEPT,records,verify,clock,save,media
from cuda_sol_oracle import sha

def read(name):
    p=ROOT/name
    return json.loads(p.read_text()) if p.exists() else None

def main():
    identity=verify();rows=records();byid={r['id']:r for r in rows};checks={};render=[]
    retry=read('resume-retest.json') or {};resumed_id=retry.get('resumed','E-resumed-sol')
    for row in rows:
        r=dict(row);path=ROOT/'runs'/r['id'];c=r.get('case')
        if not c and r['id'] in ('E-resumed-sol',resumed_id,'E-decode'):
            c=dict(byid['E-uninterrupted-sol']['case'],id=r['id'])
            r.update(case=c,mode='sol provenance / '+('resume' if 'resumed' in r['id'] else 'decode without SOL'),minimum=None)
        if not c:continue
        log=(path/'stderr.log').read_text(errors='replace')
        configurations=re.findall(r'h3(?:cli)?: DiT attention=sol ([^\r\n]+)',log)
        if configurations:
            policy={k:float(v) if any(ch in v for ch in '.eE') else int(v) for k,v in re.findall(r'(\w+)=([-+0-9.eE]+)',configurations[-1])}
            r['effective_policy']=policy;r['recorded_minimum']=r.get('minimum');r['minimum']=policy['min_exact']
        m=r.get('media') or read('runs/'+r['id']+'/media.json') or media(path,c)
        v=next((s for s in m.get('probe',{}).get('streams',[]) if s.get('codec_type')=='video'),{})
        a=next((s for s in m.get('probe',{}).get('streams',[]) if s.get('codec_type')=='audio'),{})
        expected=c['frames']-(39 if c.get('continue_from') else 0)
        strict=dict(full_decode=bool(m.get('valid')),frames=int(v.get('nb_read_frames',0))==expected,
                    geometry=(v.get('width'),v.get('height'))==(c['width'],c['height']),
                    fps=Fraction(v.get('avg_frame_rate','0'))==24,
                    audio_rate=a.get('sample_rate')=='32000',audio_channels=a.get('channels')==2,
                    video_duration=abs(float(v.get('duration',0))-expected/24)<.002,
                    audio_duration=abs(float(a.get('duration',0))-expected/24)<.034,
                    file_identity=(path/'output.mp4').exists() and sha(path/'output.mp4')==m.get('sha256'))
        r['media']=m;r['strict_media']=strict;r['status']='pass' if r['status']=='pass' and all(strict.values()) else 'failed'
        if r['id']=='E-resumed-sol' and retry:r['superseded_by']=resumed_id
        render.append(r);save(path/'final-media.json',strict)
    checks['all_complete_media']=bool(render) and all(r['status']=='pass' for r in render if not r.get('superseded_by'))
    complete={r['id'] for r in render if r['status']=='pass'}
    checks['compatibility_render_coverage']=all(name in complete for name in (
        'E-continuation-default','E-continuation-sol','E-uninterrupted-sol',
        resumed_id,'E-fast-sol','E-stream-sol','E-floor-one-sol','E-decode',
        'F-nvfp4-default','F-nvfp4-sol'))
    for name in ('R1','R2','R3','R4','R5','R6'):
        pair=[r for r in render if r['case']['id']==name]
        sol=next((r for r in pair if r['mode']=='sol'),{})
        checks[name]=len(pair)==2 and all(r['status']=='pass' for r in pair) and bool(sol.get('sol_counters')) and sol['sol_counters'][-1]['approximate']>0
    operators=read('operator-final/results.json') or []
    checks['synthetic_operator']=len(operators)==35 and all(r['passed'] for r in operators)
    route_evidence=[dict(case=r['case']['id'],complete_mask_compared=bool(r['native']['routes_exported'] and 'route_match' in r),
                         complete_mask_match=r.get('route_match') if r['native']['routes_exported'] else None) for r in operators]
    replay=read('replay-review.json') or []
    checks['real_qkv_routes_and_dense_oracle']=len(replay)==2 and all(r['passed'] for r in replay)
    checks['heldout_velocity']=all(read('heldout-velocity.json')[k]['screen_pass'] for k in ('video','audio'))
    checks['statistic_recovery']=bool((read('statistic-recovery/result.json') or {}).get('passed'))
    checks['prefix_bytes']=all((read('prefix-checks-final.json') or {}).get(m,{}).get(k,False) for m in ('default','sol') for k in ('video_prefix_exact','audio_prefix_exact'))
    checks['resume_bytes']=all((read('resume-retest-equivalence.json' if retry else 'resume-equivalence.json') or {}).get(k,False) for k in ('video_exact','audio_exact'))
    resumed=(byid.get(resumed_id,{}).get('sol_counters') or [{}])[-1]
    checks['resume_absolute_policy']=resumed.get('step')==1 and resumed.get('calls')==49 and resumed.get('dense_bypass')==1
    checks['resume_conflicts']=all((read(('resume-retest-rejection-' if retry else 'resume-rejection-')+m+'.json') or {}).get('rejected',False) for m in ('conflict','default'))
    checks['non_sol_build_rejects_sol']=bool((read('no-sol-build-rejection.json') or {}).get('passed'))
    retained=read('audit/retained-identities.json') or {}
    checks['baseline_build_identity']=retained.get('default-build/bin/h3cli',{}).get('sha256')==sha('default-build/bin/h3cli')
    checks['initial_repair_source_retained']=bool(retained.get('initial_kernel_matches_pre_repair_identity'))
    model=json.loads(Path('tests/cuda_sol_manifest.json').read_text())['model']
    model_unchanged=True
    for rel,expected in (read('model-identity.json') or {}).items():
        path=Path(model)/rel
        if not path.is_file():model_unchanged=False;continue
        st=path.stat();model_unchanged &= st.st_size==expected['size'] and st.st_mtime_ns==expected['mtime_ns']
    checks['model_sizes_and_mtimes_unchanged']=model_unchanged
    for ident in ('A-build-120','A-contracts','A-container','A-default-build',
                  'B-memcheck','B-racecheck','E-context-final','E-layout-362',
                  'E-prepared-keys','E-prefix-contract','F-interface-fp8','F-interface-nvfp4'):
        checks[ident]=byid.get(ident,{}).get('status')=='pass'
    context=ROOT/'runs/E-context-final/stderr.log'
    checks['still_no_sparsity']=context.exists() and bool(re.search(r'SOL counters .*calls=1 .*approximate=0 ',context.read_text()))
    r1a=byid.get('R1-default',{});r1b=byid.get('R1-sol',{})
    checks['primary_performance']=bool(checks['R1'] and r1a['step_seconds'][-1]/r1b['step_seconds'][-1]>=1.1 and r1b['wall_seconds']<=r1a['wall_seconds']*1.05)
    checks['workspace_limit']=all(c.get('reserved',0)<=536870912 for r in rows for c in r.get('sol_counters',[]))
    live=[]
    for p in ROOT.glob('runs/*/owner.json'):
        owner=json.loads(p.read_text())
        try:
            proc=psutil.Process(owner['pid'])
            if proc.pid!=os.getpid() and abs(proc.create_time()-owner['start_epoch'])<2:live.append(dict(record=str(p),pid=proc.pid,command=proc.cmdline()))
        except psutil.NoSuchProcess:pass
    for r in rows:
        for pid in r.get('telemetry',{}).get('owned_child_pids',[]):
            try:
                proc=psutil.Process(pid)
                if r['start_epoch']-2<=proc.create_time()<=r['finished_epoch']+1:
                    live.append(dict(record=r['id'],pid=pid,command=proc.cmdline(),status=proc.status()))
            except psutil.NoSuchProcess:pass
    checks['owned_children_exited']=not live
    checks['per_run_gpu_release']=all(r.get('gpu_released',True) for r in rows)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
    checks['final_gpu_idle']=not gpu
    limits=json.loads(ACCEPT.read_text())['buckets_minutes'];spent={b:sum(r['wall_seconds'] for r in rows if r['bucket']==b) for b in limits}
    elapsed=time.time()-clock()['start_epoch'];checks['global_deadline']=elapsed<480*60
    checks['bucket_caps']=all(spent[b]<=limits[b]*60 for b in spent)
    intervals=sorted((r['start_epoch'],r['finished_epoch']) for r in rows);covered=0.;end=0.
    for first,last in intervals:
        covered+=max(0.,last-max(first,end));end=max(end,last)
    overlap=sum(last-first for first,last in intervals)-covered
    steps=[]
    for r in rows:
        argv=r['argv']
        if '--steps' in argv:steps.append(dict(id=r['id'],steps=int(argv[argv.index('--steps')+1])))
    checks['two_step_limit']=bool(steps) and all(x['steps']<=2 for x in steps)
    # The resume uses the saved two-step schedule; its command intentionally has
    # no generation flags. The exact-state check above proves the resumed result.
    failures=[dict(id=r['id'],returncode=r['returncode'],reason=r.get('termination_reason'),log='runs/'+r['id']+'/stderr.log') for r in rows if r['status']!='pass']
    result=dict(checks=checks,passed=all(checks.values()),elapsed_seconds_at_audit=elapsed,
                buckets_child_wall_seconds=spent,recorded_process_interval_union_seconds=covered,
                overlapping_process_interval_seconds=overlap,unattributed_elapsed_seconds=elapsed-covered,
                accounting='Bucket totals are recorded child wall time. The global clock also includes verification, orchestration, media checks, failures, overlapping CPU work and final delivery.',
                live_owned_processes=live,current_gpu_processes=gpu,failures_retained=failures,resume_retest=retry,explicit_generation_steps=steps,
                synthetic_route_evidence=route_evidence,
                auxiliary_binary_sha256={name:sha(name) for name in ('default-build/bin/h3cli','bin/cuda_sol_context','bin/cuda_sol_route_probe','bin/cuda_sol_prefix','bin/cuda_sol_fingerprint','bin/quant_native','bin/sol_layout_tests','bin/sampler_tests') if Path(name).is_file()},
                route_evidence_note='The original oracle leaves route_match=true when a large case cannot export a complete mask. That placeholder is not a mask comparison; use this ledger and replay-review.json.',
                frozen_identity=identity['identity'],validation_source_sha256={str(p):sha(p) for p in Path('tests').iterdir() if p.is_file() and (p.name.startswith(('cuda_sol','test_cuda_sol')) or p.name in ('test_sol_layout.c','test_sampler.c','test_sampler_file.py','test_preview_vae.c')) and p.suffix in ('.py','.c','.cu','.json')},
                limitations=['One GPU, one sample per complete render; provisional speedups.',
                             'Two steps do not establish final visual or audio quality.',
                             'No Sage+SOL combination or external CUDA/Metal qualification.',
                             'One final bounded routing window per real-QKV capture; short synthetic cases compare full masks. The large synthetic case compares sampled operator outputs across all heads.',
                             'NVFP4 cache preparation is cold in the first smoke arm, reused in the second; no matched speed claim.',
                             'Decode and delivery are reported together after denoising; no isolated VAE kernel timing.'])
    save(ROOT/'report-renders.json',render);save(ROOT/'coverage.json',result)
    print(json.dumps(checks,indent=2));assert result['passed'],'Closing gate failed; inspect coverage.json'
if __name__=='__main__':main()
