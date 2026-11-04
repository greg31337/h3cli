#!/usr/bin/env python3
"""Bounded real-model qualification of native and packaged --render.

Each segment uses two evaluations; the three-segment continuation chain uses
256x256 / 56 frames, followed by one 256x128 / 22-frame independent shot.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

SCRIPT = '''### opening
#size 256x256
#frames 56
#steps 2
#seed 42
#offline
non_diegetic_music:
Quiet piano music.
overall_soundscape:
Soft water and a light breeze.
detailed_description:
A fixed camera watches a red toy boat on a quiet pond.
retention_analysis:
Keep the boat shape and sail color consistent.
summary:
The boat rests beside the reeds.
subject_definitions:
A red toy boat with a white sail.
### drifting
#continue
summary:
The boat drifts away from the reeds.
### farther
#continue
#continue-mode bridge
#continue-bridge-steps 1
summary:
The boat reaches the center of the pond.
### independent
#size 256x128
#frames 22
A blue toy boat on a calm pond. Soft water sounds.
'''


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def run(command, env, log, cwd):
    with log.open('wb') as f:
        process = subprocess.Popen([str(x) for x in command], env=env, cwd=cwd,
                                   stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = process.wait(timeout=1200)
            if status:
                raise subprocess.CalledProcessError(status, command)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--ffmpeg', type=Path)
    parser.add_argument('--ffprobe', type=Path)
    parser.add_argument('--packaged', action='store_true')
    parser.add_argument('--ssd-streaming', action='store_true', help='Use existing Metal SSD weight streaming')
    args = parser.parse_args()
    binary, model, out = args.binary.resolve(), args.model.resolve(), args.out.resolve()
    controls = ['--ssd-streaming'] if args.ssd_streaming else []
    out.mkdir(parents=True)
    env = dict(os.environ)
    env.update(H3_TEST_MAX_EVALUATIONS='6', H3_OFFLINE='1')
    for key in ('H3CLI_BUNDLE_INFO', 'H3_SHADER_PATH'):
        env.pop(key, None)
    if args.packaged:
        empty_path = out/'empty-path'
        empty_path.mkdir()
        env['PATH'] = str(empty_path)
        env['H3CLI_RUNTIME_CACHE'] = str(out/'runtime-cache')
        for key in ('H3_FFMPEG', 'H3_FFPROBE', 'H3_SGLANG_INPUT_FFMPEG'):
            env.pop(key, None)
    else:
        for key, value in (('H3_FFMPEG', args.ffmpeg), ('H3_FFPROBE', args.ffprobe)):
            if value:
                env[key] = str(value.resolve())
    source = out/'scenes.txt'
    source.write_text(SCRIPT)
    records = dict(passed=False, packaged=args.packaged, ssd_streaming=args.ssd_streaming, binary_sha256=sha(binary),
                   script_sha256=sha(source), cases=[])
    started = time.monotonic()
    try:
        run([binary, '--render', source, '-d', model, '--dry-run', *controls], env, out/'plan.json', out)
        plan = json.loads((out/'plan.json').read_text())
        assert [s['delivered'] for s in plan['segments']] == [56,17,17,22]
        run([binary, '--render', source, '-d', model, *controls], env, out/'render.log', out)
        result = json.loads((out/'scenes-render/run.json').read_text())
        assert result['status'] == 'complete' and all(s['status'] == 'complete' for s in result['segments'])
        assert [s['prompt'] for s in result['segments']] == [s['prompt'] for s in plan['segments']]
        assert 'subject_definitions:\nA red toy boat' in result['segments'][2]['prompt']
        assert 'retention_analysis:' not in result['segments'][3]['prompt']
        assert [(m['name'],m['frames']) for m in result['merged']] == [('opening-merged.mp4',90),('scenes.mp4',112)]
        assert result['merged'][1]['lossless_normalization']
        probe = args.ffprobe.resolve() if args.ffprobe else Path(result['tools']['ffprobe'])
        for row in [*result['segments'], *result['merged']]:
            path = Path(row['video'])
            # Independent verification, using explicit tools even when the CLI PATH is empty.
            media = json.loads(subprocess.check_output([str(probe),'-v','error','-count_frames','-show_streams','-of','json',str(path)]))
            video = next(s for s in media['streams'] if s['codec_type']=='video')
            audio = next(s for s in media['streams'] if s['codec_type']=='audio')
            frames = row.get('delivered', row['frames'])
            assert int(video['nb_read_frames']) == frames and video['avg_frame_rate']=='24/1'
            assert abs(float(audio['duration'])-frames/24)<.1
        records['cases'].append(dict(name='mixed-script', passed=True, frames=112,
                                     output_sha256=sha(out/'scenes-render/scenes.mp4')))
        disabled = out/'disabled.txt'
        disabled.write_text('### only\n#size 256x256\n#frames 22\n#steps 2\n#offline\nA red toy boat. Quiet water.\n#nomerge\n')
        run([binary,'--render',disabled,'-d',model,*controls],env,out/'nomerge.log',out)
        result = json.loads((out/'disabled-render/run.json').read_text())
        assert result['status']=='complete' and result['nomerge'] and not result['merged']
        assert sorted(p.name for p in (out/'disabled-render').glob('*.mp4')) == ['only.mp4']
        records['cases'].append(dict(name='global-nomerge',passed=True,frames=22))
        assert sha(binary) == records['binary_sha256'], 'Tested executable changed during the run'
        records['passed'] = True
    finally:
        records['elapsed_seconds'] = time.monotonic()-started
        (out/'result.json').write_text(json.dumps(records,indent=2)+'\n')
    print('PASS: real rendering, hard/bridge continuation, prompt inheritance, all-segment merge and #nomerge')


if __name__ == '__main__':
    main()
