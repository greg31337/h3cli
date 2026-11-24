#!/usr/bin/env python3
"""Bounded real-model #upscale checks: 256 -> 512, <=56 frames, two steps.

Use --backend metal for one independent K=0 clip; CUDA also exercises hard and
bridge continuations and independent K=2 refinement. Writes only under --out.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('metal', 'cuda'), required=True)
    parser.add_argument('--binary', type=Path, default=ROOT/'bin/h3cli')
    parser.add_argument('--checker', type=Path, default=ROOT/'bin/upscale_av_tests')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--models-path', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary, checker = args.binary.resolve(), args.checker.resolve()
    # Native test admission accepts budgets 6 or 50; commands below explicitly
    # use two generation steps and at most two refinement evaluations.
    env = dict(os.environ, H3_OFFLINE='1', H3_TEST_MAX_EVALUATIONS='6')
    for key in ('H3_REFERENCE_MEDIA', 'H3_REFERENCE_MODEL', 'H3_REFERENCE_REGRESSION_OUT'):
        env.pop(key, None)
    ffprobe = env.get('H3_FFPROBE', 'ffprobe')
    ffmpeg = env.get('H3_FFMPEG', 'ffmpeg')
    common = [str(binary), '--models-path', str(args.models_path.resolve()), '-d', str(args.model.resolve()),
              '--offline', '--verbose', '--steps', '2', '--width', '256', '--height', '256']
    reference = ROOT/'samples/panda/panda.png'
    prompt = 'A black and white cat walks slowly beside a garden. Soft birdsong and a gentle breeze.'
    head = f'#ref-image "{reference}"\n'
    cases = {'independent-zero': '#upscale 0\n#nomerge\n### one\n#frames 22\n'+prompt}
    if args.backend == 'cuda':
        cases = {mode: '#upscale 0\n'+head+'### one\n#frames 39\n'+prompt+
                 f'\n### two\n#continue\n#continue-mode {mode}\n#continue-bridge-steps 1\n#frames 56\n'+prompt
                 for mode in ('hard', 'bridge')}
        cases['refine-two'] = '#upscale 2\n'+head+'### one\n#frames 22\n'+prompt+'\n### two\n'+prompt
    results = {'backend': args.backend, 'binary_sha256': sha(binary), 'cases': {}}
    record = out/'results.json'
    for name, text in cases.items():
        script = out/(name+'.txt')
        script.write_text(text+'\n')
        destination = out/name
        command = common+['--render', str(script), '--output-dir', str(destination)]
        # Assert the generated workload, including private full-context targets,
        # before any expensive command. No arbitrary geometry overrides.
        planned = subprocess.run(command+['--dry-run'], env=env, capture_output=True, text=True)
        (out/(name+'-plan.log')).write_text(planned.stderr)
        assert planned.returncode == 0, planned.stderr
        plan = json.loads(planned.stdout)
        for segment in plan['segments']:
            assert (segment['source_width'],segment['source_height'],segment['width'],segment['height']) == (256,256,512,512)
            assert 22 <= segment['frames'] <= 56
            assert segment['command'][segment['command'].index('--steps')+1] == '2'
        assert plan['upscale_refine_steps'] in (0,2)
        assert sum(s['delivered'] for s in plan['segments']) <= 56
        (out/(name+'-plan.json')).write_text(json.dumps(plan, indent=2)+'\n')
        started = time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        row = {'command': command, 'exit_code': completed.returncode, 'seconds': time.monotonic()-started, 'passed': False}
        results['cases'][name] = row
        record.write_text(json.dumps(results, indent=2)+'\n')
        assert completed.returncode == 0, f'{name}: inspect {out/(name+".log")}'
        report = json.loads((destination/'run.json').read_text())
        assert report['status'] == 'complete'
        row['media'] = []
        for segment in report['segments']:
            assert segment['upscale_status'] == 'complete'
            assert Path(segment['source_video']).is_file()
            if plan['upscale_refine_steps'] == 0:
                subprocess.run([str(checker), '--compare', segment['state'], segment['upscaled_state']], env=env, check=True)
                worker_log = (destination/'.render'/(Path(segment['state']).stem+'-upscale.log')).read_text()
                assert 'no DiT evaluations' in worker_log
                assert 'denoise ' not in worker_log
            if segment['continuing']:
                command = segment['command']
                previous = report['segments'][report['segments'].index(segment)-1]['state']
                assert command[command.index('--continue-from')+1] == previous
        for media in [*report['segments'], *report['merged']]:
            path = Path(media['video'])
            frames = media['delivered'] if 'delivered' in media else media['frames']
            assert frames <= 56
            probe = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(path)], env=env))
            video = next(s for s in probe['streams'] if s['codec_type']=='video')
            audio = next(s for s in probe['streams'] if s['codec_type']=='audio')
            assert (int(video['width']), int(video['height']), int(video['nb_read_frames'])) == (512,512,frames)
            assert video['r_frame_rate'] == '24/1' and abs(float(audio['duration'])-frames/24) < .1
            subprocess.run([ffmpeg, '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], env=env, check=True)
            row['media'].append({'path': str(path), 'frames': frames, 'sha256': sha(path)})
        row['passed'] = True
        record.write_text(json.dumps(results, indent=2)+'\n')
        print(f'{name}: passed ({row["seconds"]:.2f}s)', flush=True)


if __name__ == '__main__':
    main()
