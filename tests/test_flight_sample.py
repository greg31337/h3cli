#!/usr/bin/env python3
"""Check flight planning and assembly without loading or downloading H3 models.

Run: python3 tests/test_flight_sample.py
The integration check substitutes only H3 inference; FFmpeg/FFprobe are real.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT/'samples/flight'
spec = importlib.util.spec_from_file_location('flight_sample', SAMPLE/'test.py')
flight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flight)


class FlightSampleTests(unittest.TestCase):
    def plan(self, *options):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(['sh', str(SAMPLE/'run.sh'), '--dry-run', *options],
                                    cwd=directory, capture_output=True, text=True, check=True)
            self.assertEqual(list(Path(directory).iterdir()), [])
            return json.loads(result.stdout)

    def test_route_and_default_plan(self):
        plan = self.plan()
        self.assertEqual((plan['width'], plan['height']), (960, 544))
        self.assertEqual((plan['output_width'], plan['output_height']), (1920, 1088))
        self.assertEqual(plan['delivered_frames'], 4841)
        rows = plan['segments']
        self.assertEqual([row['number'] for row in rows], list(map(str, range(1, 21))))
        for index, row in enumerate(rows):
            render, upscale = row['render_command'], row['upscale_command']
            self.assertNotIn('--continue-from', render)
            self.assertIn('--state-only', render)
            self.assertEqual(render[render.index('--width')+1], '960')
            self.assertEqual(render[render.index('--height')+1], '544')
            self.assertEqual(render[render.index('--seed')+1], str(42+index))
            self.assertEqual(upscale[upscale.index('--upscale-state')+1], row['source'])
            self.assertEqual(upscale[upscale.index('--upscale-refine-steps')+1], '4')
            for conflicting in ('-p', '--width', '--height', '--frames', '--steps', '--first-frame'):
                self.assertNotIn(conflicting, upscale)
            if index:
                prior = rows[index-1]
                self.assertEqual(render[render.index('--first-frame')+1], prior['handoff'])
                prior_end = prior['prompt'].split('End state: ')[1].split('\n')[0]
                self.assertIn(prior_end, row['prompt'])
                self.assertEqual(row['frames'], 242)
            else:
                self.assertNotIn('--first-frame', render)
                self.assertEqual(row['frames'], 243)

    def test_transfer_only_and_model_paths(self):
        plan = self.plan('--upscale-refine-steps', '0', '--steps', '2',
                         '--models-path', '/models with spaces', '--offline',
                         '--upscale-model', '/weights with spaces/upscale.safetensors')
        for row in plan['segments']:
            upscale = row['upscale_command']
            self.assertNotIn('--upscale-noise', upscale)
            self.assertEqual(upscale[upscale.index('--upscale-refine-steps')+1], '0')
            self.assertEqual(upscale[upscale.index('--upscale-model')+1],
                             '/weights with spaces/upscale.safetensors')
            for command in (row['render_command'], upscale):
                self.assertIn('--offline', command)
                self.assertEqual(command[command.index('--models-path')+1], '/models with spaces')

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires FFmpeg and FFprobe')
    def test_real_media_handoff_and_assembly(self):
        with tempfile.TemporaryDirectory(prefix='flight sample ') as directory:
            work = Path(directory)
            fixture = work/'fixture.mp4'
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error',
                            '-f', 'lavfi', '-i', 'color=c=navy:s=1920x1088:r=24:d=10.125',
                            '-f', 'lavfi', '-i', 'sine=frequency=220:sample_rate=32000:duration=10.125',
                            '-vf', "drawbox=x=0:y=0:w=64:h=64:color=white:t=fill:enable='eq(n,242)'",
                            '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
                            '-c:a', 'aac', '-ac', '2', str(fixture)], check=True)
            fake = work/'fake-h3cli'
            fake.write_text(f'#!{sys.executable}\n'+'''import json
import os
from pathlib import Path
import shutil
import sys
args = sys.argv[1:]
def value(flag):
    return args[args.index(flag)+1]
if '--save-upscale-state' in args:
    assert '--state-only' in args and '--continue-from' not in args
    if '--first-frame' in args:
        assert Path(value('--first-frame')).read_bytes().startswith(b'\\x89PNG')
    Path(value('--save-upscale-state')).write_text(json.dumps(args))
else:
    assert Path(value('--upscale-state')).is_file()
    assert '--width' not in args and '-p' not in args
    shutil.copyfile(os.environ['FLIGHT_TEST_FIXTURE'], value('-o'))
    state = value('--save-av-state')
    Path(state).write_bytes(b'synthetic state')
    Path(state+'.presentation').write_text('synthetic presentation')
''')
            fake.chmod(0o755)
            prompts = work/'two.txt'
            prompts.write_text('### 1 ###\nFly forward.\n### 2 ###\nKeep flying forward.\n')
            output = work/'combined.mp4'
            states = work/'states'
            command = [sys.executable, str(SAMPLE/'test.py'), str(prompts),
                       '--h3cli', str(fake), '--no-overlay', '--offline',
                       '--work-dir', str(states), '-o', str(output)]
            env = dict(os.environ, FLIGHT_TEST_FIXTURE=str(fixture))
            result = subprocess.run(command, env=env, cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            flight.inspect_video('ffprobe', output, 485)
            report = json.loads((states/'run.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['delivered_frames'], 485)
            self.assertEqual(report['segments'][1]['first_frame'], report['segments'][0]['handoff'])
            # The white marker exists only on the final frame, so a first-frame
            # or approximate seek extraction cannot pass this pixel check.
            pixel = subprocess.check_output(['ffmpeg', '-v', 'error', '-i',
                                             report['segments'][0]['handoff'], '-vf',
                                             'crop=1:1:10:10', '-pix_fmt', 'rgb24',
                                             '-f', 'rawvideo', '-'])
            self.assertEqual(len(pixel), 3)
            self.assertTrue(all(channel > 240 for channel in pixel), pixel)
            before = output.read_bytes()
            retry = subprocess.run(command, env=env, cwd=work, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn('Output already exists', retry.stderr)
            self.assertEqual(output.read_bytes(), before)
            # Exercise the badge path as well as the default trim path.
            overlay = work/'overlay-preflight'
            overlay.mkdir()
            flight.preflight('ffmpeg', 'ffprobe', overlay, overlay=True)


if __name__ == '__main__':
    unittest.main()
