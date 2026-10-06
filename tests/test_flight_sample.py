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
        self.assertEqual((plan['output_width'], plan['output_height']), (960, 544))
        self.assertEqual(plan['context'], 39)
        self.assertEqual(plan['handoff_mode'], 'av-continuation')
        self.assertEqual(plan['delivered_frames'], 10239)
        self.assertEqual(plan['duration_seconds'], 426.625)
        rows = plan['segments']
        self.assertEqual([row['number'] for row in rows], list(map(str, range(1, 51))))
        for index, row in enumerate(rows):
            render = row['render_command']
            for forbidden in ('--ref-image', '--first-frame', '--state-only',
                              '--save-upscale-state', '--upscale-state'):
                self.assertNotIn(forbidden, render)
            self.assertNotIn('upscale_command', row)
            self.assertNotIn('handoff_command', row)
            self.assertEqual(render[render.index('--width')+1], '960')
            self.assertEqual(render[render.index('--height')+1], '544')
            self.assertEqual(render[render.index('--frames')+1], '243')
            self.assertEqual(render[render.index('--steps')+1], '50')
            self.assertEqual(render[render.index('--seed')+1], str(42+index))
            self.assertEqual(render[render.index('--save-av-state')+1], row['state'])
            self.assertEqual(render[render.index('-o')+1], row['video'])
            self.assertIn('--no-preview-vae', render)
            prepare = row['prepare_command']
            self.assertEqual(prepare[prepare.index('-c:v')+1], 'copy')
            self.assertNotIn('start_frame', ' '.join(prepare))
            self.assertNotIn('start_sample', ' '.join(prepare))
            if index:
                prior = rows[index-1]
                self.assertEqual(row['continue_from'], prior['state'])
                self.assertEqual(render[render.index('--continue-from')+1], prior['state'])
                self.assertEqual(render[render.index('--continue-context')+1], '39')
                prior_end = prior['prompt'].split('End state: ')[1].split('\n')[0]
                self.assertIn('Begin in the inherited ending view: '+prior_end, row['prompt'])
                self.assertEqual(row['frames'], 204)
            else:
                self.assertIsNone(row['continue_from'])
                self.assertNotIn('--continue-from', render)
                self.assertNotIn('--continue-context', render)
                self.assertEqual(row['frames'], 243)

    def test_model_paths_and_steps(self):
        plan = self.plan('--steps', '2', '--seed', '90',
                         '-d', '/main model', '--models-path', '/models with spaces', '--offline')
        for index, row in enumerate(plan['segments']):
            command = row['render_command']
            self.assertIn('--offline', command)
            self.assertEqual(command[command.index('--steps')+1], '2')
            self.assertEqual(command[command.index('--seed')+1], str(90+index))
            self.assertEqual(command[command.index('-d')+1], '/main model')
            self.assertEqual(command[command.index('--models-path')+1], '/models with spaces')

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires FFmpeg and FFprobe')
    def test_real_media_continuation_and_assembly(self):
        with tempfile.TemporaryDirectory(prefix='flight sample ') as directory:
            work = Path(directory)
            for name, frames in [('source', 243), ('continuation', 204)]:
                command = ['ffmpeg', '-nostdin', '-v', 'error',
                           '-f', 'lavfi', '-i', f'color=c=navy:s=960x544:r=24:d={frames/24}',
                           '-f', 'lavfi', '-i', f'sine=frequency=220:sample_rate=32000:duration={frames/24}']
                if name == 'continuation':
                    command += ['-vf', "drawbox=x=0:y=0:w=64:h=64:color=white:t=fill:enable='eq(n,0)'"]
                subprocess.run(command + ['-c:v', 'libx264', '-preset', 'ultrafast',
                               '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ac', '2',
                               str(work/(name+'.mp4'))], check=True)
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
shutil.copyfile(Path(os.environ['FLIGHT_TEST_FIXTURES'])/(name+'.mp4'), value('-o'))
state = value('--save-av-state')
Path(state).write_text(json.dumps(args))
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
            env = dict(os.environ, FLIGHT_TEST_FIXTURES=str(work))
            result = subprocess.run(command, env=env, cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            flight.inspect_video('ffprobe', output, 447)
            report = json.loads((states/'run.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['delivered_frames'], 447)
            self.assertEqual([row['frames'] for row in report['segments']], [243, 204])
            self.assertEqual(report['segments'][1]['continue_from'], report['segments'][0]['state'])
            # The first NEW continuation frame carries this marker. Double
            # trimming context (or one old handoff frame) would remove it.
            pixel = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output),
                                             '-vf', 'select=eq(n\\,243),format=rgb24,crop=1:1:10:10',
                                             '-frames:v', '1', '-pix_fmt', 'rgb24',
                                             '-f', 'rawvideo', '-'])
            self.assertEqual(len(pixel), 3)
            self.assertTrue(all(channel > 240 for channel in pixel), pixel)
            before = output.read_bytes()
            retry = subprocess.run(command, env=env, cwd=work, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn('Output already exists', retry.stderr)
            self.assertEqual(output.read_bytes(), before)
            # Exercise the badge path as well as the default stream-copy path.
            overlay = work/'overlay-preflight'
            overlay.mkdir()
            flight.preflight('ffmpeg', 'ffprobe', overlay, overlay=True)


if __name__ == '__main__':
    unittest.main()
