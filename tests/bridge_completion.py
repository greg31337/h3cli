#!/usr/bin/env python3
"""Combine T065's independent gates; never promote correctness to quality."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def sidecar(base,ext): return Path(str(base)+'.'+ext)


def verified_runs(directory):
    files=list(directory.glob('*-run.json')); assert files
    for file in files:
        run=read(file); assert run['passed'] and set(run['variants'])==set(run['outputs'])
        source=Path(run['source']); assert sha(sidecar(source,'h3av'))==run['source_sha256']
        for name,outputs in run['outputs'].items():
            for ext,value in outputs.items(): assert sha(sidecar(directory/name,ext))==value
    return True


def hard_regression(directory):
    report=read(directory/'results.json')
    assert report['passed'] and report['steps']==20 and report['baseline_commit'].startswith('6c09804')
    runs=report['runs']
    assert runs['source-before']['hashes']==runs['source-after']['hashes']
    assert runs['hard-before']['hashes']==runs['hard-default']['hashes']==runs['hard-explicit']['hashes']
    for name,run in runs.items():
        for ext,value in run['hashes'].items(): assert sha(sidecar(directory/name,ext))==value
    return True


def unmasked_regression(directory):
    report=read(directory/'results.json')
    for name in ['t2va','fl2va']:
        case=report[name]; assert case['passed'] and case['original']['hashes']==case['current']['hashes']
        for label in ['original','current']:
            for ext,value in case[label]['hashes'].items(): assert sha(sidecar(directory/f'{name}-{label}',ext))==value
    return True


def quality(directory):
    assert verified_runs(directory)
    report=read(directory/'quality.json')
    protocol=Path(__file__).resolve().parents[1]/'docs/bridge-quality-protocol.json'
    assert report['protocol_sha256']==sha(protocol)
    assert report['complete'] and report['quality_accepted'] and report['quality_gate']['passed'], report['quality_gate']['problems']
    assert sha(directory/'selection.json')==report['selection_sha256']
    assert sha(directory/'visual-review.json')==report['visual_review_sha256']
    return True


def samplers(directory):
    assert verified_runs(directory)
    report=read(directory/'sampler-results.json'); assert report['passed']
    assert len(report['comparisons'])==7
    for case in report['comparisons'].values():
        for stream in ['video','audio']: assert case[stream]['bit_identical']
        if 'trajectory' in case: assert case['trajectory']['bit_identical']
        assert all(case['decoded_outputs_identical'].values())
    return True


def chain_visual_review(boundaries, reviews):
    fields=['scene','background','character','color','camera',
            'action_adherence','body_integrity','transition']
    for boundary in boundaries:
        name=Path(boundary['target']).name
        review=reviews[name]
        assert review.get('reviewer') and review.get('method') and review.get('notes')
        for role in ['source','target']:
            assert review[role+'_rgb_sha256']==sha(sidecar(Path(boundary[role]),'rgb'))
        assert all(review['checks'].get(field)=='pass' for field in fields), name
    return True


def chain(directory):
    assert verified_runs(directory)
    report=read(directory/'chain.json'); assert report['passed'] and report['segments']>=3
    boundaries=report['boundaries']; assert len(boundaries)>=2
    assert len(boundaries)+1==report['segments']
    references={json.dumps(read(path)['references'],sort_keys=True) for path in directory.glob('*-run.json')}
    assert len(references)==1
    assert len({b['prompt'] for b in boundaries})==len(boundaries)
    for i,boundary in enumerate(boundaries):
        for role in ['source','target']:
            assert sha(sidecar(Path(boundary[role]),'h3av'))==boundary[role+'_sha256']
        if i: assert boundary['source']==boundaries[i-1]['target']
    assert read(directory/'quality.json')['complete']
    assert chain_visual_review(boundaries,read(directory/'visual-review.json'))
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,nargs='?',default=Path('outputs/bridge-quality'))
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--chain-directory',default='chain',help='Chain result directory, relative to the result root or absolute')
    args=parser.parse_args(); root=args.directory.resolve(); checks={}
    for label,function,folder in [('hard_regression',hard_regression,'hard-regression'),
                                  ('normal_generation_regression',unmasked_regression,'unmasked-regression'),
                                  ('same_scene_motion_quality',quality,'acceptance'),
                                  ('cpu_gpu_reuse_exact_av',samplers,'samplers'),
                                  ('multi_segment_chain',chain,args.chain_directory)]:
        try:
            checks[label]=dict(passed=function(root/folder))
        except (AssertionError,FileNotFoundError,KeyError,ValueError) as error:
            checks[label]=dict(passed=False,reason=str(error) or 'required evidence failed verification')
    report=dict(complete=all(c['passed'] for c in checks.values()),checks=checks,
                chain_directory=str((root/args.chain_directory).resolve()),
                policy='Bridge remains opt-in and T065 remains open unless every gate passes.')
    root.mkdir(parents=True,exist_ok=True)
    (root/'completion.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if args.require_complete: assert report['complete'], 'Bridge completion gate failed; inspect completion.json'


if __name__=='__main__': main()
