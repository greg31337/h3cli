#!/usr/bin/env python3
"""Validate independent upscale planning, cropping, timing and quiet clip boundaries.

Only H3 inference is substituted in the integration test; FFmpeg/FFprobe are real.
"""
from array import array
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT/'samples/miniatures'
spec = importlib.util.spec_from_file_location('miniatures_sample', SAMPLE/'test.py')
miniatures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(miniatures)


class MiniaturesSampleTests(unittest.TestCase):
    def plan(self, *options):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(['sh', str(SAMPLE/'run.sh'), '--dry-run', *options],
                                    cwd=directory, capture_output=True, text=True, check=True)
            self.assertEqual(list(Path(directory).iterdir()), [])
            return json.loads(result.stdout)

    def test_independent_plan(self):
        plan = self.plan()
        self.assertEqual((plan['width'], plan['height']), (960, 544))
        self.assertEqual((plan['upscale_width'], plan['upscale_height']), (1920, 1088))
        self.assertEqual((plan['output_width'], plan['output_height']), (1920, 1080))
        self.assertTrue(plan['independent_segments'])
        self.assertEqual((plan['delivered_frames'], plan['duration_seconds']), (9600, 400))
        rows = plan['segments']
        self.assertEqual([r['number'] for r in rows], list(map(str, range(1, 41))))
        self.assertEqual(len({r['source'] for r in rows}), 40)
        self.assertEqual(len({r['state'] for r in rows}), 40)
        for index, row in enumerate(rows):
            render, upscale, prepare = row['render_command'], row['upscale_command'], row['prepare_command']
            self.assertEqual(row['frames'], 240)
            for command in (render, upscale):
                for flag in ('--continue-from', '--continue-context', '--bridge-from',
                             '--first-frame', '--last-frame', '--ref-image', '--ref-video'):
                    self.assertNotIn(flag, command)
            self.assertEqual(render[render.index('--width')+1], '960')
            self.assertEqual(render[render.index('--height')+1], '544')
            self.assertEqual(render[render.index('--frames')+1], '243')
            self.assertEqual(render[render.index('--steps')+1], '50')
            self.assertEqual(render[render.index('--seed')+1], str(42+index))
            self.assertIn('--state-only', render)
            self.assertIn('--no-preview-vae', render)
            self.assertEqual(render[render.index('--save-upscale-state')+1], row['source'])
            self.assertEqual(upscale[upscale.index('--upscale-state')+1], row['source'])
            self.assertEqual(upscale[upscale.index('--save-av-state')+1], row['state'])
            self.assertEqual(upscale[upscale.index('--upscale-seed')+1], str(42+index))
            self.assertEqual(upscale[upscale.index('--upscale-refine-steps')+1], '4')
            self.assertEqual(upscale[upscale.index('--upscale-noise')+1], '0.25')
            for flag in ('-p', '--width', '--height', '--frames', '--steps'):
                self.assertNotIn(flag, upscale)
            filters = prepare[prepare.index('-filter_complex')+1]
            self.assertIn('trim=end_frame=240', filters)
            self.assertIn('crop=1920:1080:0:4', filters)
            self.assertNotIn('scale=', filters)
            self.assertIn('first and last seconds are nearly silent', row['prompt'])

    def test_transfer_only_and_model_paths(self):
        plan = self.plan('--steps', '2', '--upscale-refine-steps', '0', '--seed', '90',
                         '--models-path', '/models with spaces', '-d', '/main model',
                         '--upscale-model', '/weights with spaces/upscale.safetensors', '--offline')
        for index, row in enumerate(plan['segments']):
            render, upscale = row['render_command'], row['upscale_command']
            for command in (render, upscale):
                self.assertIn('--offline', command)
                self.assertEqual(command[command.index('--models-path')+1], '/models with spaces')
                self.assertEqual(command[command.index('-d')+1], '/main model')
            self.assertEqual(render[render.index('--steps')+1], '2')
            self.assertEqual(render[render.index('--seed')+1], str(90+index))
            self.assertEqual(upscale[upscale.index('--upscale-refine-steps')+1], '0')
            self.assertNotIn('--upscale-noise', upscale)
            self.assertEqual(upscale[upscale.index('--upscale-model')+1],
                             '/weights with spaces/upscale.safetensors')

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires FFmpeg and FFprobe')
    def test_real_media_crop_duration_and_audio(self):
        with tempfile.TemporaryDirectory(prefix='miniatures sample ') as directory:
            work = Path(directory)
            fixture = work/'native.mp4'
            # White bars occupy ONLY the four rows to be cropped from each edge.
            # Green final frames must be trimmed off, never shown at a join.
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error',
                            '-f', 'lavfi', '-i', 'color=c=navy:s=1920x1088:r=24:d=10.125',
                            '-f', 'lavfi', '-i', 'sine=frequency=1000:sample_rate=32000:duration=10.125',
                            '-vf', 'drawbox=x=0:y=0:w=iw:h=4:color=white:t=fill,'
                                   'drawbox=x=0:y=1084:w=iw:h=4:color=white:t=fill,'
                                   "drawbox=x=0:y=0:w=iw:h=ih:color=lime:t=fill:enable='gte(n,240)'",
                            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0',
                            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ac', '2', str(fixture)], check=True)
            fake = work/'fake-h3cli'
            fake.write_text(f'#!{sys.executable}\n'+'''import json
import os
from pathlib import Path
import shutil
import sys
args = sys.argv[1:]
def value(flag):
    return args[args.index(flag)+1]
assert not any(flag in args for flag in ('--continue-from', '--continue-context',
                                        '--first-frame', '--last-frame', '--ref-image'))
if '--save-upscale-state' in args:
    assert '--state-only' in args and value('--width') == '960' and value('--height') == '544'
    assert value('--frames') == '243'
    Path(value('--save-upscale-state')).write_text(json.dumps(args))
else:
    assert '-p' not in args and '--width' not in args and '--frames' not in args
    source = json.loads(Path(value('--upscale-state')).read_text())
    assert value('--upscale-seed') == source[source.index('--seed')+1]
    shutil.copyfile(os.environ['MINIATURES_TEST_FIXTURE'], value('-o'))
    state = value('--save-av-state')
    Path(state).write_text('synthetic upscaled state')
    Path(state+'.presentation').write_text('synthetic presentation')
''')
            fake.chmod(0o755)
            prompts = work/'two.txt'
            prompts.write_text('### 1 ###\nA tiny brass city.\n### 2 ###\nA paper garden.\n')
            output, states = work/'combined.mp4', work/'states'
            command = [sys.executable, str(SAMPLE/'test.py'), str(prompts),
                       '--h3cli', str(fake), '--no-overlay', '--offline',
                       '--work-dir', str(states), '-o', str(output)]
            env = dict(os.environ, MINIATURES_TEST_FIXTURE=str(fixture))
            result = subprocess.run(command, env=env, cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads((states/'run.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['delivered_frames'], 480)
            self.assertEqual(report['duration_seconds'], 20)
            miniatures.inspect_video('ffprobe', output, 480)
            for row in report['segments']:
                miniatures.inspect_video('ffprobe', row['video'], 243, 1920, 1088)
                miniatures.inspect_video('ffprobe', row['prepared'], 240)

            # Check both crop edges and both sides of the final cut in decoded pixels.
            for frame, y in ((0, 1), (0, 1078), (239, 500), (240, 500), (479, 500)):
                pixel = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output),
                    '-vf', f'select=eq(n\\,{frame}),format=rgb24,crop=1:1:100:{y}',
                    '-frames:v', '1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'])
                self.assertEqual(len(pixel), 3)
                self.assertLess(pixel[0], 25)
                self.assertLess(pixel[1], 25)
                self.assertGreater(pixel[2], 80)

            def pcm(path):
                raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path),
                    '-map', '0:a:0', '-ac', '1', '-ar', '32000', '-f', 's16le', '-'])
                data = array('h', raw)
                if sys.byteorder != 'little':
                    data.byteswap()
                return data

            def rms(data):
                return math.sqrt(sum(v*v for v in data)/len(data))

            original = pcm(fixture)
            prepared = pcm(report['segments'][0]['prepared'])
            self.assertEqual(len(prepared), 320000)
            middle = rms(prepared[160000:163200])
            self.assertAlmostEqual(middle/rms(original[160000:163200]), 0.5, delta=0.02)
            self.assertLess(rms(prepared[:320]), middle*0.04)
            self.assertLess(rms(prepared[-320:]), middle*0.04)
            merged = pcm(output)
            self.assertLess(abs(len(merged)-640000), 1024)
            self.assertLess(rms(merged[319680:320320]), middle*0.05)

            before = output.read_bytes()
            retry = subprocess.run(command, env=env, cwd=work, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn('Output already exists', retry.stderr)
            self.assertEqual(output.read_bytes(), before)
            overlay = work/'overlay-preflight'
            overlay.mkdir()
            miniatures.preflight('ffmpeg', 'ffprobe', overlay, overlay=True)


if __name__ == '__main__':
    unittest.main()
