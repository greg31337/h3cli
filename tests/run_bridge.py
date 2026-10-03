#!/usr/bin/env python3
"""Real-model bridge integration matrix for T021-T042. Run GPU jobs sequentially."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from run_continuation import prepare_fixtures

ROOT = Path(__file__).resolve().parents[1]
WALK = ('The woman with the face in <Picture 1> and the outfit in <Picture 2> '
        'walks slowly through a sunlit garden and waves. Steady camera and quiet outdoor ambience.')
POSE = ('The woman with the face in <Picture 1> and the outfit in <Picture 2> '
        'completes her walking step, gradually turns toward the camera, and raises both arms. '
        'Sunlit garden, steady camera, quiet outdoor ambience.')
RUN = ('The woman with the face in <Picture 1> and the outfit in <Picture 2> '
       'gently lowers her arms, turns left, and begins jogging along the garden path. '
       'Sunlit garden, steady camera, quiet outdoor ambience.')
MIXED = 'The woman completes her current motion and gradually turns toward the camera. Natural daylight, steady camera, quiet outdoor ambience.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT/'models/MiniMax-H3')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/bridge-integration/acceptance')
    parser.add_argument('--hard-state', type=Path, help='Optional existing hard-continuation state at 256 square, at least 39 frames')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--only', help='Comma-separated case names; dependencies must already exist')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    model = str(args.model.resolve())
    source = args.hard_state.resolve() if args.hard_state else out/'source-hard.h3av'
    fixture = ROOT/'outputs/continuation-validation/reference.mp4'
    driver = ROOT/'bin/continuation_generate'
    matrix = []

    def item(name, mode='image1', state=source, continuation='bridge', prompt=POSE,
             profile='stepped', strength=.5, bridge_steps=8, seed=44, pair=None):
        cmd = [str(driver), model, str(out/name), mode, str(state) if state else '-',
               str(seed), '90', str(args.steps), '1', '0', prompt, str(fixture), '39', '256',
               continuation, profile, str(bridge_steps), str(strength)]
        matrix.append((name, cmd, pair))

    if not args.hard_state:
        item('source', state=None, continuation='hard', prompt=WALK, seed=42)
        item('source-hard', state=out/'source.h3av', continuation='hard', prompt=WALK, seed=43)
    item('bridge-unchanged', prompt=WALK)
    item('bridge-pose')
    item('bridge-to-bridge', state=out/'bridge-pose.h3av', prompt=RUN, profile='linear', strength=.4, bridge_steps=6, seed=45)
    item('bridge-to-hard', state=out/'bridge-pose.h3av', prompt=RUN, continuation='hard', seed=45)
    item('image2', mode='image2', profile='ease-out', strength=.65, bridge_steps=9)
    item('image-video', mode='imagevideo', prompt=MIXED)
    item('image-audio', mode='imageaudio', prompt=MIXED)
    item('video-audio', mode='video', prompt=MIXED)
    item('replaced-audio', mode='replace', prompt=MIXED)
    item('t2va', mode='t2va', prompt=MIXED)
    item('trim-normal', pair=('H3_TEST_KEEP_PAIR', 'trim-debug'))
    item('zero-hard', continuation='hard', prompt=WALK, pair=('H3_TEST_ZERO_BRIDGE_PAIR', 'zero-bridge'))
    if args.only:
        wanted = set(args.only.split(',')); matrix = [m for m in matrix if m[0] in wanted]
        assert wanted == {m[0] for m in matrix}, 'unknown case'
    if args.list:
        print('\n'.join(m[0] for m in matrix)); return
    prepare_fixtures()
    subprocess.run(['make','-j8','all','bin/continuation_generate'], cwd=ROOT, check=True)
    binary_hash = sha(driver)
    records_path = out/'runs.json'
    records = json.loads(records_path.read_text()) if args.resume and records_path.exists() else {}
    env = {k:v for k,v in os.environ.items() if not k.startswith('H3_')}
    # Keep the integration suite on the CPU oracle. GPU selection and parity
    # have dedicated coverage in run_bridge_quality.py --phase samplers.
    env.update(H3_PROFILE='1', H3_GPU_SAMPLER='1', H3_CPU_SAMPLER='1')
    for name, cmd, pair in matrix:
        source_hash = sha(cmd[4]) if cmd[4] != '-' else None
        reference_hashes = {p.name:sha(p) for p in (ROOT/'inputs').glob('*.jpg')}
        previous = records.get(name,{})
        if (args.resume and previous.get('returncode') == 0 and previous.get('command') == cmd
                and previous.get('binary_sha256') == binary_hash and previous.get('source_sha256') == source_hash
                and previous.get('references_sha256') == reference_hashes
                and all((out/(case+'.'+ext)).exists() and sha(out/(case+'.'+ext)) == hashes[ext]
                        for case, hashes in previous.get('outputs',{}).items() for ext in hashes)):
            continue
        run_env = dict(env)
        if pair: run_env[pair[0]] = str(out/pair[1])
        print('running',name,flush=True); start=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=run_env,stdout=log,stderr=log)
        record={'command':cmd,'returncode':result.returncode,'seconds':time.monotonic()-start,
                'binary_sha256':binary_hash,'source_sha256':source_hash,'references_sha256':reference_hashes,'outputs':{}}
        if result.returncode == 0:
            for case in [name]+([pair[1]] if pair else []):
                record['outputs'][case]={ext:sha(out/(case+'.'+ext)) for ext in ['h3av','mp4','rgb','pcm','json']}
            if source_hash: assert sha(cmd[4]) == source_hash, 'source state was modified'
        records[name]=record
        temporary=records_path.with_suffix('.tmp'); temporary.write_text(json.dumps(records,indent=2)+'\n'); temporary.replace(records_path)
        print(name,result.returncode,round(record['seconds'],2),flush=True)
        if result.returncode:
            print((out/(name+'.log')).read_text()[-4000:]); raise SystemExit(result.returncode)
    print('ok: requested bridge integration renders completed',flush=True)


if __name__ == '__main__': main()
