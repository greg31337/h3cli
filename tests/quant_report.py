#!/usr/bin/env python3
"""Summarize retained evidence without launching GPU work or changing its ledger."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from quant_followup import decisions as followup_decisions


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def profiles(text):
    result=[]
    pattern=(r'h3(?:cli)?: CUDA profile (.*?) / (.*?): wall ([\d.]+)s peak ([\d.]+) GiB, '
             r'GEMM (\d+) attention (\d+) conv (\d+), H2D ([\d.]+) GiB D2H ([\d.]+) GiB '
             r'stream ([\d.]+) GiB copy ([\d.]+)s wait ([\d.]+)s\n'
             r'h3(?:cli)?: CUDA category seconds: GEMM ([\d.]+) attention ([\d.]+) conv ([\d.]+)')
    keys=('wall_seconds','peak_owned_gib','gemm_calls','attention_calls','conv_calls',
          'h2d_gib','d2h_gib','streamed_gib','copy_wall_seconds','wait_wall_seconds',
          'gemm_event_seconds','attention_event_seconds','conv_event_seconds')
    for match in re.finditer(pattern,text):
        result.append(dict(component=match[1],phase=match[2],**dict(zip(keys,map(float,match.groups()[2:])))))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('outputs/quant-5090'))
    parser.add_argument('--publish',type=Path,help='optional compact tracked evidence index')
    args=parser.parse_args();out=args.out
    ledger=json.loads((out/'budget.json').read_text())
    assert not ledger.get('pending') and (ledger['limit'] is None or ledger['seconds']<=ledger['limit'])
    acceptance=json.loads((out/'quality/acceptance.json').read_text())
    assert acceptance['gallery_manifest_sha256']==sha(out/'quality/manifest.json')
    assert acceptance['profile_sha256']==hashlib.sha256(json.dumps(acceptance['profile'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    quality=json.loads((out/'quality/manifest.json').read_text())
    for case in quality:
        assert set(case['outputs'])=={'off','fp8','nvfp4'}
        for artifact in case['outputs'].values():
            assert sha(out/'quality'/Path(artifact['path']).name)==artifact['sha256']
    for mode,decision in acceptance['decisions'].items():
        assert decision['status']=='accepted'
        for case in quality:
            assert sha(out/'quality'/Path(case['outputs'][mode]['path']).name)==decision['media'][case['name']]
    result=dict(schema=1,date=acceptance['date'],starting_revision=json.loads((out/'environment.json').read_text())['starting_revision'],
        evidence_root=str(out),budget=dict(seconds=ledger['seconds'],limit=ledger['limit'],extensions=ledger.get('extensions',[])),
        quality=acceptance,artifacts={},runs=[],warm={},primary={},projections=[],models={})
    for path in [out/'budget.json',out/'environment.json',out/'media-index.json',out/'quality/manifest.json',
                 out/'quality/acceptance.json',out/'quality/review.html',out/'continuation-joins.json',out/'continuation-review.html']:
        result['artifacts'][str(path.relative_to(out))]=sha(path)
    result['builds']={p.stem:json.loads(p.read_text()) for p in out.glob('*.identity.json')}
    for row in ledger['runs']:
        name=row['name'];record=out/(name+'.json');log=out/(name+'.log')
        entry={k:row.get(k) for k in ('name','status','seconds','timeout','failure','binary_sha256')}
        entry.update(record=str(record.relative_to(out)),record_sha256=sha(record),log=str(log.relative_to(out)),log_sha256=sha(log))
        result['runs'].append(entry)
        text=log.read_text(errors='replace')
        if name.startswith('projection-'):
            result['projections'] += [dict(case=name,**json.loads(line)) for line in text.splitlines() if line.startswith('{')]
        if name.startswith('warm'):
            measurements={}
            for match in re.finditer(r'PASS (first|warm): ([\d.]+)s denoise=([\d.]+)s frames=(\d+) previews=(\d+) latent_callbacks=(\d+)',text):
                measurements[match[1]]=dict(mp4_and_state_seconds=float(match[2]),denoise_seconds=float(match[3]),
                                           frames=int(match[4]),evaluations=int(match[6])-1)
            measurements['profiles_cumulative']=profiles(text)
            measurements['quant_profiles']=[line for line in text.splitlines() if 'quant profile H3 DiT' in line]
            measurements['settings']='20 evaluations, 50 blocks, fast CUDA, tiny VAE, synchronous finite-latent callbacks, face1, seed72'
            result['warm'][name]=measurements
        if name in ('baseline-image05','fp8-image05-cold','nvfp4-image05-cold','quality-detail-off','quality-detail-fp8','quality-detail-nvfp4'):
            phases=profiles(text)
            load=next(x for x in phases if x['component']=='H3 DiT' and x['phase']=='load')
            denoise=next(x for x in phases if x['component']=='H3 DiT' and 'denoise' in x['phase'])
            result['primary'][name]=dict(denoise_seconds=denoise['wall_seconds']-load['wall_seconds'],
                mp4_seconds=float(re.search(r'total wall time: ([\d.]+) s',text)[1]),profiles_cumulative=phases,
                weight_prepare_or_verify_seconds=sum(float(x) for x in re.findall(r'quant weight .*? time=([\d.]+)s',text)),
                prepared=sum(' prepared bytes=' in line for line in text.splitlines()),
                cache_hits=sum(' cache-hit bytes=' in line for line in text.splitlines()))
    for path in sorted((out/'quality').glob('*.presentation')):
        fields={line.split()[0]:line.split()[1:] for line in path.read_text().splitlines()}
        if 'denoise_quant' in fields:
            variant='Ref2VA' if fields['variant']==['1'] else 'FL2VA'
            result['models'].setdefault(variant,set()).add(fields['denoise_quant'][2])
    result['models']={k:sorted(v) for k,v in result['models'].items()}
    media=json.loads((out/'media-index.json').read_text())
    result['media']=dict(complete_decodes=len(media),finite_states=sum(x.get('finite_latents',False) for x in media),
                         black_flags=[x['path'] for x in media if x['black_intervals']])
    result['host_logs']={str(p.relative_to(out)):sha(p) for p in sorted((out/'local').glob('*.log'))}
    result['coverage_limits']=[
        'One warmed repeat per mode/preset, no long/high-resolution render or other-GPU qualification.',
        'Initial pilot records predate per-run source inventories; immutable binary hashes, build IDs and final-state model fingerprints provide retrospective links.'
    ]
    follow=out/'followup'
    groups=dict(
        standalone_preparation=[f'followup-prepare-{m}' for m in ('fp8','nvfp4')],
        model_variants=[f'followup-variants-fixedplans-{m}' for m in ('fp8','nvfp4')],
        adapter_contexts=[f'followup-adapters-fixedplans-{m}' for m in ('fp8','nvfp4')],
        pre_vae_reclamation=[f'followup-reclaim-{m}' for m in ('fp8','nvfp4')],
        sanitizers=[f'followup-{tool}-{kind}-{m}' for m in ('fp8','nvfp4')
                    for tool,kind in [('memcheck','lifecycle'),('memcheck','pressure'),('synccheck','lifecycle'),('racecheck','lifecycle')]],
        turbo=[f'followup-turbo-1-{m}' for m in ('off','fp8','nvfp4')]+['followup-cache-identities','followup-turbo-05-nvfp4']+
              [f'followup-scale-{s}-{m}' for s in ('1','05') for m in ('fp8','nvfp4')],
        continuation=[f'followup-seam-{m}-{kind}' for m in ('off','fp8','nvfp4') for kind in ('source','hard','bridge')]+['followup-review'])
    by_name={run['name']:run for run in ledger['runs']}
    result['followup']=dict(checks={name:dict(passed=all(by_name.get(case,{}).get('status')=='passed' for case in cases),cases=cases)
                                           for name,cases in groups.items()})
    result['followup']['resolved_test_failures']={
        f'followup-{kind}-fp8':dict(repeat=f'followup-{kind}-fixedplans-fp8',
            resolved=by_name.get(f'followup-{kind}-fixedplans-fp8',{}).get('status')=='passed',
            reason='Exact-repeat identity assertion requires fixed GEMM selection when conditioning is rebuilt.')
        for kind in ('adapters','variants') if by_name.get(f'followup-{kind}-fp8',{}).get('status')=='failed'}
    if (follow/'review-manifest.json').exists():
        result['followup']['quality']=followup_decisions(follow)
    for name,check in result['followup']['checks'].items():
        if not check['passed']:result['coverage_limits'].append(f'Follow-up coverage incomplete: {name}.')
    for scope in ('continuation','turbo'):
        if result['followup'].get('quality',{}).get(scope,{}).get('status')!='accepted':
            result['coverage_limits'].append(f'Human playback/listening acceptance pending for the new {scope} outputs.')
    result['coverage_limits'] += [
        'Model directories are immutable per context; adapter changes were tested through separate contexts and a return to the original context.',
        'Follow-up synccheck/racecheck filter owned quant_ kernels; full memcheck includes unfiltered lifecycle and pressure execution.',
        'Turbo scale 1.0 has full off/FP8/NVFP4 render coverage; scale 0.5 has full NVFP4 generation and four-family FP8/NVFP4 operator/cache checks.'
    ]
    for path in sorted(follow.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.html','.log','.txt') and not path.name.startswith('.'):
            result['artifacts'][str(path.relative_to(out))]=sha(path)
    result['decision']={mode:dict(status='go for opt-in base-model 5090 previews',quality='accepted',
        denoise_speedups={str(size):result['warm'][f'warm{size}-{baseline}']['warm']['denoise_seconds']/result['warm'][f'warm{size}-{mode}']['warm']['denoise_seconds'] for size in (288,480)},
        baseline=baseline) for mode,baseline in (('fp8','off'),('nvfp4','fp8'))}
    (out/'validation-index.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.publish:
        compact={k:v for k,v in result.items() if k not in ('runs','warm','primary','projections','host_logs')}
        compact['index_sha256']=sha(out/'validation-index.json')
        compact['index_path']=str(out/'validation-index.json')
        compact['warm']={k:{n:v[n] for n in ('first','warm','settings')} for k,v in result['warm'].items()}
        compact['failures']=[r for r in result['runs'] if r['status']!='passed']
        compact['run_count']=len(result['runs'])
        args.publish.write_text(json.dumps(compact,indent=2)+'\n')
    print('Wrote validation-index.json; both quality decisions match exact media hashes.')


if __name__=='__main__':main()
