#!/usr/bin/env python3
"""Check the native flight and panda samples without loading H3 models.

Run: python3 tests/test_flight_sample.py
Build bin/h3cli and bin/render_driver first. The assembly check substitutes
only H3 inference; FFmpeg/FFprobe are real.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get('H3_RENDER_BIN', ROOT/'bin/h3cli')).resolve()
DRIVER = Path(os.environ.get('H3_RENDER_DRIVER', ROOT/'bin/render_driver')).resolve()
FFMPEG = shutil.which(os.environ.get('H3_FFMPEG', 'ffmpeg'))
FFPROBE = shutil.which(os.environ.get('H3_FFPROBE', 'ffprobe'))


def value(command, flag):
    return command[command.index('--'+flag)+1]


class RenderSampleTests(unittest.TestCase):
    def setUp(self):
        self.env = dict(os.environ, H3CLI_BIN=str(BINARY))
        self.env.pop('H3_TEST_MAX_EVALUATIONS', None)
        self.env.pop('H3_MODEL_DIR', None)

    def plan(self, sample='flight', *options):
        with tempfile.TemporaryDirectory(prefix="sample launch's ") as directory:
            result = subprocess.run(['sh', str(ROOT/'samples'/sample/'run.sh'),
                                     '--dry-run', *options], cwd=directory,
                                    env=self.env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])
            plan = json.loads(result.stdout)
            expected = Path(directory).resolve()/f'outputs/{sample}-render'
            if '--output-dir' in options:
                expected = Path(directory).resolve()/value(list(options), 'output-dir')
            self.assertEqual(Path(plan['output']), expected)
            return plan

    def check_sequence(self, plan, sample, count):
        self.assertFalse(plan['overlay_segment_name'])
        self.assertEqual(plan['seed_set'], [])
        rows = plan['segments']
        self.assertEqual([row['label'] for row in rows], list(map(str, range(1, count+1))))
        self.assertEqual(sum(row['delivered'] for row in rows), 243+(count-1)*204)
        self.assertEqual([merge['name'] for merge in plan['merged']], ['1-merged.mp4', sample+'.mp4'])
        for merge in plan['merged']:
            self.assertEqual(merge['segments'], [row['name'] for row in rows])
        for index, row in enumerate(rows):
            command = row['command']
            self.assertEqual((row['width'], row['height'], row['frames']), (960, 544, 243))
            for flag, expected in [('width', '960'), ('height', '544'), ('frames', '243'),
                                   ('steps', '50'), ('seed', '42')]:
                self.assertEqual(value(command, flag), expected)
            self.assertEqual(row['seed'], 42)
            self.assertIn('--no-preview-vae', command)
            self.assertEqual(Path(row['video']).name, f'{index+1}.mp4')
            for forbidden in ('--first-frame', '--state-only', '--save-upscale-state', '--upscale-state'):
                self.assertNotIn(forbidden, command)
            self.assertEqual(row['continuing'], bool(index))
            if index:
                self.assertEqual(value(command, 'continue-from'), value(rows[index-1]['command'], 'save-av-state'))
                self.assertEqual(value(command, 'continue-context'), '39')
                self.assertEqual(row['delivered'], 204)
            else:
                self.assertNotIn('--continue-from', command)
                self.assertNotIn('--continue-context', command)
                self.assertEqual(row['delivered'], 243)

    def test_route_and_default_plan(self):
        plan = self.plan()
        self.check_sequence(plan, 'flight', 50)
        self.assertEqual(plan['references'], {})
        for index, row in enumerate(plan['segments']):
            self.assertNotIn('--ref-image', row['command'])
            if index:
                prior_end = plan['segments'][index-1]['prompt'].split('End state: ')[1].split('\n')[0]
                self.assertIn('Begin in the inherited ending view: '+prior_end, row['prompt'])

    def test_panda_reference_in_every_segment(self):
        plan = self.plan('panda')
        self.check_sequence(plan, 'panda', 40)
        reference = str(ROOT/'samples/panda/panda.png')
        self.assertEqual(plan['references'], {'panda': reference})
        for row in plan['segments']:
            self.assertEqual(value(row['command'], 'ref-image'), reference)
            self.assertEqual(value(row['command'], 'ref-image-size'), 'high')

    def test_model_paths_launcher_and_script_settings(self):
        with tempfile.TemporaryDirectory(prefix="sample binary's ") as directory:
            wrapper = Path(directory)/'selected h3cli'
            marker = Path(directory)/'called'
            wrapper.write_text(f'#!{sys.executable}\nimport os, pathlib, sys\n'
                               f'pathlib.Path({str(marker)!r}).touch()\n'
                               f'os.execv({str(BINARY)!r}, [{str(BINARY)!r}, *sys.argv[1:]])\n')
            wrapper.chmod(0o755)
            self.env['H3CLI_BIN'] = str(wrapper)
            plan = self.plan('flight', '--steps', '2', '--seed', '90',
                             '-d', '/main model', '--models-path', '/models with spaces',
                             '--offline', '--output-dir', 'another output')
            self.assertTrue(marker.is_file())
        for row in plan['segments']:
            command = row['command']
            self.assertIn('--offline', command)
            # Explicit script settings take precedence over CLI defaults.
            self.assertEqual(value(command, 'steps'), '50')
            self.assertEqual(value(command, 'seed'), '42')
            self.assertEqual(value(command, 'model-dir'), '/main model')
            self.assertEqual(value(command, 'models-path'), '/models with spaces')

    @unittest.skipUnless(FFMPEG and FFPROBE, 'requires FFmpeg and FFprobe')
    def test_real_media_continuation_and_assembly(self):
        with tempfile.TemporaryDirectory(prefix="flight sample's ") as directory:
            work = Path(directory).resolve()
            for name, frames in [('source', 243), ('continuation', 204)]:
                command = [FFMPEG, '-nostdin', '-v', 'error',
                           '-f', 'lavfi', '-i', f'color=c=navy:s=960x544:r=24:d={frames/24}',
                           '-f', 'lavfi', '-i', f'sine=frequency=220:sample_rate=32000:duration={frames/24}']
                if name == 'continuation':
                    command += ['-vf', "drawbox=x=0:y=0:w=64:h=64:color=white:t=fill:enable='eq(n,0)'"]
                subprocess.run(command + ['-c:v', 'libx264', '-preset', 'ultrafast',
                                         '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ac', '2',
                                         str(work/(name+'.mp4'))], check=True, timeout=60)
            fake = work/'fake-h3cli'
            fake.write_text(f'#!{sys.executable}\n'+'''import json
import os
from pathlib import Path
import shutil
import sys
args = sys.argv[1:]
def value(flag):
    return args[args.index(flag)+1]
assert not any(flag in args for flag in ('--ref-image', '--first-frame', '--state-only',
                                        '--save-upscale-state', '--upscale-state'))
name = 'source'
if '--continue-from' in args:
    previous = Path(value('--continue-from'))
    assert previous.is_file() and Path(str(previous)+'.presentation').is_file()
    assert value('--continue-context') == '39'
    assert '--save-av-state' in json.loads(previous.read_text())
    name = 'continuation'
shutil.copyfile(Path(os.environ['FLIGHT_TEST_FIXTURES'])/(name+'.mp4'), value('--output'))
state = value('--save-av-state')
Path(state).write_text(json.dumps(args))
Path(state+'.presentation').write_text('synthetic presentation')
''')
            fake.chmod(0o755)
            # Render the actual first two sample segments through production
            # planning and media assembly, substituting only the worker.
            text = (ROOT/'samples/flight/flight.txt').read_text()
            third = re.search(r'^### 3\s*$', text, re.MULTILINE)
            self.assertIsNotNone(third)
            script = work/'two.txt'
            script.write_text(text[:third.start()])
            output = work/'render'
            movie = output/'two.mp4'
            command = [str(DRIVER), str(fake), '--render', str(script),
                       '--offline', '--output-dir', str(output)]
            env = dict(self.env, FLIGHT_TEST_FIXTURES=str(work), H3_FFMPEG=FFMPEG, H3_FFPROBE=FFPROBE)
            result = subprocess.run(command, env=env, cwd=work, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            metadata = json.loads(subprocess.check_output([FFPROBE, '-v', 'error', '-count_frames',
                                  '-show_streams', '-of', 'json', str(movie)], text=True, timeout=30))
            video = next(stream for stream in metadata['streams'] if stream['codec_type'] == 'video')
            self.assertEqual((video['width'], video['height'], int(video['nb_read_frames'])), (960, 544, 447))
            audio = next(stream for stream in metadata['streams'] if stream['codec_type'] == 'audio')
            self.assertAlmostEqual(float(audio['duration']), 447/24, delta=.1)
            report = json.loads((output/'run.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual([row['delivered'] for row in report['segments']], [243, 204])
            self.assertTrue(all(row['status'] == 'complete' for row in report['merged']))
            self.assertEqual(value(report['segments'][1]['command'], 'continue-from'),
                             report['segments'][0]['state'])
            # The first NEW continuation frame carries this marker. Trimming
            # context twice (or repeating an old frame) would remove it.
            pixel = subprocess.check_output([FFMPEG, '-v', 'error', '-i', str(movie),
                                             '-vf', 'select=eq(n\\,243),format=rgb24,crop=1:1:10:10',
                                             '-frames:v', '1', '-pix_fmt', 'rgb24',
                                             '-f', 'rawvideo', '-'], timeout=30)
            self.assertEqual(len(pixel), 3)
            self.assertTrue(all(channel > 240 for channel in pixel), pixel)
            before = movie.read_bytes()
            retry = subprocess.run(command, env=env, cwd=work, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(movie.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
