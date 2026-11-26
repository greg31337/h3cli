#!/usr/bin/env python3
"""Production native runner with synthetic workers and real FFmpeg/FFprobe.

No GPU or models required. --out retains fixtures, media, logs and results.
The injected worker exists only in bin/render_driver, not in production h3cli.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path(os.environ.get('H3_RENDER_DRIVER', ROOT/'bin/render_driver')).resolve()
OUTPUT = None


class NativeMediaTests(unittest.TestCase):
    def setUp(self):
        if OUTPUT:
            self.root = OUTPUT / self._testMethodName
            self.root.mkdir()
        else:
            temporary = tempfile.TemporaryDirectory(prefix="h3 native media's ")
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name).resolve()
        self.out = self.root/'output'
        self.source = self.root/'scenes.txt'
        self.env = dict(os.environ)
        self.env.pop('H3_RENDER_SCAN_TIMEOUT', None)
        self.env.pop('H3_RENDER_VERIFY_FRAMES', None)
        self.ffmpeg = str(Path(shutil.which(self.env.get('H3_FFMPEG', 'ffmpeg'))).resolve())
        self.ffprobe = str(Path(shutil.which(self.env.get('H3_FFPROBE', 'ffprobe'))).resolve())
        self.env.update(H3_FFMPEG=self.ffmpeg, H3_FFPROBE=self.ffprobe)
        self.worker = self.root/'synthetic worker.py'
        self.worker.write_text(f'''#!{sys.executable}
import hashlib, json, math, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
def value(name, default=None):
    return args[args.index('--'+name)+1] if '--'+name in args else default
prompt = value('prompt', '')
upscale_source = value('upscale-av-state') or value('upscale-state')
saved = json.loads(Path(upscale_source).read_text()) if upscale_source else None
if saved:
    prompt = saved['prompt']
    assert '--prompt' not in args and '--continue-from' not in args
    if value('upscale-av-state'):
        assert value('upscale-refine-steps') == '0'
    if 'FAIL_UPSCALE' in prompt:
        print('synthetic upscale failure', file=sys.stderr)
        raise SystemExit(7)
if 'RACE_OUTPUT' in prompt:
    (Path(value('output')).parent.parent/'a.mp4').write_bytes(b'concurrent output; preserve me')
if 'FAIL_WORKER' in prompt:
    print('synthetic worker failure', file=sys.stderr)
    raise SystemExit(7)
frames = int(value('frames', '56'))
if '--seconds' in args:
    frames = math.floor(float(value('seconds'))*24+.5)
frames = max(frames, 5) + (5-max(frames, 5))%17
trim = 0
if '--continue-from' in args:
    assert Path(value('continue-from')).is_file()
    trim = int(value('continue-context', '39'))
width, height = int(value('width','864')), int(value('height','480'))
if saved:
    width, height, frames, trim = saved['width']*2, saved['height']*2, saved['frames'], saved['trim']
metadata = dict(width=width, height=height, frames=frames, trim=trim, prompt=prompt)
references = [Path(args[i+1]) for i,a in enumerate(args[:-1]) if a == '--ref-image']
for reference in references:
    assert reference.is_file(), reference
    if '-ref-' in reference.name:
        assert reference.read_bytes().startswith(b'\\x89PNG'), reference
metadata['reference_sha256'] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in references]
if '--continue-from' in args:
    previous = json.loads(Path(value('continue-from')).read_text())
    assert (previous['width'], previous['height']) == (width, height)
frames -= trim
color = 'blue' if 'blue' in prompt else 'red'
frequency = 660 if color == 'blue' else 440
encoding = ['-c:v', 'libx264rgb', '-crf', '0', '-pix_fmt', 'rgb24'] if '--lossless-video' in args else ['-c:v','libx264','-crf','18','-pix_fmt','yuv420p']
visual = f'color=c={{color}}:s={{width}}x{{height}}:r=24:d={{frames/24:.12f}}'
if 'FRAME_PATTERN' in prompt:
    visual = f'testsrc2=s={{width}}x{{height}}:r=24:d={{frames/24:.12f}}'
command = [os.environ['H3_FFMPEG'], '-nostdin','-v','error','-n', '-f','lavfi','-i',
           visual,
           '-f','lavfi','-i',f'sine=frequency={{frequency}}:sample_rate=32000','-t',f'{{frames/24:.12f}}',
           '-frames:v',str(frames),*encoding,'-c:a','aac','-ar','32000','-ac','2',value('output')]
subprocess.run(command, check=True)
state = Path(value('save-av-state'))
state.write_text(json.dumps(metadata))
state.with_suffix('.argv.json').write_text(json.dumps(args))
Path(str(state)+'.presentation').write_text('Synthetic presentation.\\n')
if value('save-upscale-state'):
    Path(value('save-upscale-state')).write_text(json.dumps(metadata))
with (state.parent/'worker-order.jsonl').open('a') as log:
    log.write(json.dumps(dict(stage='upscale' if saved else 'generation', state=state.name))+'\\n')
''')
        self.worker.chmod(0o755)

    def command(self, text):
        self.source.write_text(text)
        return [str(DRIVER), str(self.worker), '--render', str(self.source), '--output-dir', str(self.out)]

    def run_script(self, text, expected=0):
        result = subprocess.run(self.command(text), env=self.env, capture_output=True, text=True, timeout=60)
        (self.root/'driver.log').write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, expected, result.stderr)
        return json.loads((self.out/'run.json').read_text())

    def rgb(self, path, canvas=None):
        command = [self.ffmpeg, '-v', 'error', '-i', str(path), '-map', '0:v:0']
        if canvas:
            command += ['-vf', f'scale={canvas[0]}:{canvas[1]}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={canvas[0]}:{canvas[1]}:(ow-iw)/2:(oh-ih)/2,setsar=1']
        return subprocess.check_output(command + ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])

    def test_previous_frames_and_independent_prompt_inheritance(self):
        report = self.run_script('''#seed-set 1,2
### a
#size 256x256
#frames 22
#steps 2
subject_definitions:
A red cat.
summary:
FRAME_PATTERN
non_diegetic_music:
N/A
### b
#use-last-sections
#ref-image #last-segment-frame 1
#ref-image #last-segment-frame -1
summary:
blue
### c
#use-last-sections
#ref-image #last-segment-frame -10
summary:
red
''')
        self.assertEqual(report['status'], 'complete')
        for start in (0,3):
            first,second,third = report['segments'][start:start+3]
            pixels = self.rgb(first['video']); stride = 256*256*3
            self.assertNotEqual(pixels[:stride], pixels[-stride:])
            for row,source in ((second,first),(third,second)):
                self.assertFalse(row['continuing'])
                self.assertIn('subject_definitions:\nA red cat.', row['prompt'])
                self.assertNotIn('--continue-from', row['command'])
                source_pixels = self.rgb(source['video'])
                hashes = []
                for ref in row['frame_references']:
                    self.assertEqual(ref['status'], 'complete')
                    self.assertEqual(ref['source_video'], source['video'])
                    index=ref['frame_index']
                    self.assertEqual(self.rgb(ref['image']), source_pixels[index*stride:(index+1)*stride])
                    hashes.append(hashlib.sha256(Path(ref['image']).read_bytes()).hexdigest())
                self.assertEqual(json.loads(Path(row['state']).read_text())['reference_sha256'], hashes)
        self.assertEqual([m['frames'] for m in report['merged']], [66,66])

    def test_previous_frame_after_upscale_before_overlay(self):
        report = self.run_script('''#upscale 0
#overlay-segment-name
### a
#size 256x256
#frames 22
#steps 2
FRAME_PATTERN
### b
#ref-image #last-segment-frame -1
blue
''')
        first,second = report['segments']; ref=second['frame_references'][0]
        self.assertEqual(ref['source_video'], first['original_video'])
        self.assertTrue(ref['source_video'].endswith('0000-upscaled.mp4'))
        frame = self.rgb(ref['image']); stride=512*512*3
        self.assertEqual(frame, self.rgb(first['original_video'])[-stride:])
        self.assertNotEqual(frame, self.rgb(first['video'])[-stride:])
        self.assertEqual(report['merged'][0]['frames'], 44)

    def test_previous_frame_from_continuation_delivered_suffix(self):
        report = self.run_script('''### a
#size 256x256
#frames 39
#steps 2
red
### b
#continue
#frames 56
FRAME_PATTERN
### c
#frames 22
#ref-image #last-segment-frame -17
#ref-image #last-segment-frame -1
blue
''')
        first,second,third=report['segments']; stride=256*256*3
        self.assertEqual(second['delivered'],17)
        pixels=self.rgb(second['video'])
        self.assertEqual(len(pixels),17*stride)
        self.assertEqual([r['frame_index'] for r in third['frame_references']],[0,16])
        for ref,expected in zip(third['frame_references'],(pixels[:stride],pixels[-stride:])):
            self.assertEqual(self.rgb(ref['image']),expected)

    def test_reference_extraction_failure_stops_before_next_worker(self):
        wrapper = self.root/'failing extraction.py'
        wrapper.write_text(f'''#!{sys.executable}
import os,sys
if '-c:v' in sys.argv and sys.argv[sys.argv.index('-c:v')+1]=='png':
    print('injected extraction failure',file=sys.stderr)
    raise SystemExit(7)
os.execv({self.ffmpeg!r},[{self.ffmpeg!r},*sys.argv[1:]])
''')
        wrapper.chmod(0o755);self.env['H3_FFMPEG']=str(wrapper)
        report=self.run_script('### a\n#size 256x256\n#frames 22\n#steps 2\nred\n'
                               '### b\n#ref-image #last-segment-frame -1\nblue',1)
        self.assertEqual([r['status'] for r in report['segments']],['complete','failed'])
        self.assertEqual(report['segments'][1]['stage'],'reference extraction')
        self.assertEqual(report['segments'][1]['frame_references'][0]['status'],'failed')
        self.assertTrue((self.out/'a.mp4').exists())
        self.assertFalse((self.out/'.render/0001.h3av').exists())
        self.assertFalse((self.out/'scenes.mp4').exists())

    def probe_wrapper(self, edit='', on_scan='pass'):
        wrapper = self.root/'probe wrapper.py'
        wrapper.write_text(f'''#!{sys.executable}
import json, signal, subprocess, sys, time
from pathlib import Path
root = Path({str(self.root)!r})
args = sys.argv[1:]
name = Path(args[-1]).name
scan = '-count_frames' in args
with (root/'probe-calls.jsonl').open('a') as log:
    log.write(json.dumps(dict(name=name, scan=scan, args=args))+'\\n')
if scan:
{textwrap.indent(on_scan, '    ')}
data = json.loads(subprocess.check_output([{self.ffprobe!r}, *args], text=True))
{edit}
print(json.dumps(data))
''')
        wrapper.chmod(0o755)
        self.env['H3_FFPROBE'] = str(wrapper)

    def probe_calls(self):
        return [json.loads(line) for line in (self.root/'probe-calls.jsonl').read_text().splitlines()]

    def test_upscale_continuation_overlay_and_seed_runs(self):
        report = self.run_script('''#upscale 0
#overlay-segment-name
#seed-set 0,1
### a
#size 256x256
#frames 39
red
### b
#continue
#frames 56
blue
''')
        self.assertEqual(report['status'], 'complete')
        self.assertEqual([m['frames'] for m in report['merged']], [56,56,56,56])
        self.assertTrue(all((m['width'],m['height'])==(512,512) for m in report['merged']))
        order=[json.loads(line)['stage'] for line in (self.out/'.render/worker-order.jsonl').read_text().splitlines()]
        self.assertEqual(order, ['generation','upscale']*4)
        for offset in (0,1):
            rows=report['segments'][2*offset:2*offset+2]
            for row in rows:
                self.assertEqual(row['upscale_status'], 'complete')
                native=json.loads(Path(row['state']).read_text())
                upscaled=json.loads(Path(row['upscaled_state']).read_text())
                self.assertEqual((native['width'],native['height']), (256,256))
                self.assertEqual((upscaled['width'],upscaled['height']), (512,512))
                self.assertEqual(native['trim'], upscaled['trim'])
                self.assertTrue(Path(row['source_video']).is_file())
                original=self.rgb(Path(row['original_video']))
                labeled=self.rgb(Path(row['video']))
                self.assertNotEqual(original,labeled)
                self.assertEqual(original[:512*256*3],labeled[:512*256*3])
            second=rows[1]['command']
            self.assertEqual(second[second.index('--continue-from')+1],rows[0]['state'])
            merged=self.rgb(self.out/f'seed-{offset}/scenes.mp4')
            self.assertEqual(merged,b''.join(self.rgb(Path(row['video'])) for row in rows))
            self.assertEqual(len(merged),56*512*512*3)

    def test_upscale_refinement_capture_and_encoding_merge(self):
        report=self.run_script('#upscale 2\n### a\n#size 256x256\n#frames 22\nred\n### b\n#lossless-video\nblue')
        self.assertEqual(report['status'],'complete')
        self.assertEqual(report['merged'][0]['frames'],44)
        self.assertTrue(report['merged'][0]['lossless_normalization'])
        for i,row in enumerate(report['segments']):
            self.assertTrue((self.out/f'.render/{i:04d}.h3up').is_file())
            self.assertIn('--upscale-state',row['upscale_command'])
            self.assertEqual(row['upscale_status'],'complete')
        merged=self.rgb(self.out/'scenes.mp4')
        self.assertEqual(merged,b''.join(self.rgb(Path(row['video'])) for row in report['segments']))
        self.assertEqual(len(merged),44*512*512*3)

    def test_upscale_nomerge_still_publishes_only_upscaled_clips(self):
        report=self.run_script('#upscale 0\n#nomerge\n### a\n#size 256x256\n#frames 22\nred')
        self.assertEqual(report['merged'],[])
        self.assertEqual(sorted(p.name for p in self.out.glob('*.mp4')),['a.mp4'])
        self.assertEqual(len(self.rgb(self.out/'a.mp4')),22*512*512*3)

    def test_upscale_failure_preserves_sources_and_prevents_merging(self):
        report=self.run_script('#upscale 0\n### a\n#size 256x256\n#frames 22\nred\n### b\nFAIL_UPSCALE',1)
        self.assertEqual(report['status'],'failed')
        self.assertEqual([s['status'] for s in report['segments']],['complete','failed'])
        self.assertEqual(report['segments'][1]['upscale_status'],'failed')
        self.assertEqual(report['segments'][1]['stage'],'upscale')
        self.assertTrue((self.out/'a.mp4').is_file())
        self.assertFalse((self.out/'b.mp4').exists())
        self.assertFalse((self.out/'scenes.mp4').exists())
        self.assertTrue((self.out/'.render/0001.h3av').is_file())
        self.assertTrue((self.out/'.render/0001.mp4').is_file())

    def test_upscale_invalid_delivery_is_not_published(self):
        self.probe_wrapper(edit="""if name == '0000-upscaled.mp4':
    next(s for s in data['streams'] if s['codec_type']=='video')['width'] = 256
""")
        report=self.run_script('#upscale 0\n### a\n#size 256x256\n#frames 22\nred',1)
        self.assertEqual(report['segments'][0]['upscale_status'],'failed')
        self.assertFalse((self.out/'a.mp4').exists())
        self.assertFalse((self.out/'scenes.mp4').exists())
        self.assertTrue((self.out/'.render/0000.h3av').is_file())

    def test_upscale_cancellation_preserves_native_state(self):
        self.worker.write_text(self.worker.read_text().replace("if saved:\n    prompt", """if saved:
    import signal, time
    def stop(sig, frame):
        Path(value('output')).with_suffix('.stopped').touch()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    Path(value('output')).with_suffix('.ready').touch()
    time.sleep(30)
if saved:
    prompt"""))
        process=subprocess.Popen(self.command('#upscale 0\n### a\n#size 256x256\n#frames 22\nred'),
                                 env=self.env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            ready=self.out/'.render/0000-upscaled.ready'
            deadline=time.monotonic()+15
            while not ready.exists() and process.poll() is None and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertTrue(ready.exists(),'upscale worker did not start')
            process.send_signal(signal.SIGINT)
            output,_=process.communicate(timeout=10)
            (self.root/'driver.log').write_bytes(output)
            self.assertEqual(process.returncode,130,output.decode())
            report=json.loads((self.out/'run.json').read_text())
            self.assertEqual(report['status'],'interrupted')
            self.assertEqual(report['segments'][0]['upscale_status'],'interrupted')
            self.assertTrue((self.out/'.render/0000-upscaled.stopped').exists())
            self.assertTrue((self.out/'.render/0000.h3av').is_file())
            self.assertFalse((self.out/'a.mp4').exists())
            self.assertFalse((self.out/'scenes.mp4').exists())
        finally:
            if process.poll() is None:
                process.kill();process.wait()
            process.stdout.close()

    def test_seed_runs_aliases_overlays_and_independent_continuations(self):
        reference = self.root/'reference image.png'
        reference.touch()
        report = self.run_script('''#overlay-segment-name
#seed-set 100,200
#ref actor "reference image.png"
### A
#size 128x96
#frames 56
#ref-image actor
red
### B
#continue
#seed 2
#ref-image actor
blue
### C
#frames 22
red
''')
        self.assertEqual(report['status'], 'complete')
        self.assertTrue(report['overlay_segment_name'])
        self.assertEqual([s['seed'] for s in report['segments']], [142,102,102,242,202,202])
        self.assertEqual(list(self.out.glob('*.mp4')), [])
        for run, offset in enumerate((100,200)):
            folder = self.out/f'seed-{offset}'
            self.assertEqual(sorted(p.name for p in folder.glob('*.mp4')), ['A-merged.mp4','A.mp4','B.mp4','C.mp4','scenes.mp4'])
            clips = [self.rgb(folder/(name+'.mp4')) for name in ('A','B','C')]
            self.assertEqual(self.rgb(folder/'scenes.mp4'), b''.join(clips))
            self.assertEqual(self.rgb(folder/'A-merged.mp4'), b''.join(clips[:2]))
            for j, row in enumerate(report['segments'][run*3:run*3+3]):
                original = self.rgb(Path(row['original_video']))
                labeled = clips[j]
                self.assertNotEqual(original, labeled)
                self.assertEqual(original[:128*48*3], labeled[:128*48*3])
                self.assertIn(b'\xff\xff\xff', labeled[128*48*3:128*96*3])
                self.assertIn(b'\xff\xff\xff', labeled[-128*48*3:])
                audio_args = ['-map','0:a:0','-f','s16le','-']
                before = subprocess.check_output([self.ffmpeg,'-v','error','-i',row['original_video'],*audio_args])
                after = subprocess.check_output([self.ffmpeg,'-v','error','-i',row['video'],*audio_args])
                self.assertEqual(before, after)
                args = json.loads((self.out/f'.render/{run*3+j:04d}.argv.json').read_text())
                self.assertEqual(args[args.index('--seed')+1], str(row['seed']))
                if j==0:
                    self.assertNotIn('--continue-from', args)
                elif j==1:
                    self.assertEqual(args[args.index('--continue-from')+1], report['segments'][run*3]['state'])
                    self.assertEqual(args[args.index('--ref-image')+1], str(reference))
                else:
                    self.assertNotIn('--ref-image', args)

    def test_seed_run_completes_merges_before_next_run_fails(self):
        self.worker.write_text(self.worker.read_text().replace("prompt = value('prompt', '')",
            "if value('seed') == '242': raise SystemExit(7)\nprompt = value('prompt', '')"))
        report = self.run_script('#seed-set 100,200\n### a\n#size 96x96\n#frames 22\nred', 1)
        self.assertTrue((self.out/'seed-100/scenes.mp4').exists())
        self.assertFalse((self.out/'seed-200/scenes.mp4').exists())
        self.assertEqual([s['status'] for s in report['segments']], ['complete','failed'])
        self.assertEqual([s['status'] for s in report['merged']], ['complete','planned'])

    def test_overlay_preflight_failure_prevents_workers(self):
        wrapper = self.root/'no-overlay.py'
        wrapper.write_text(f'''#!{sys.executable}
import os,sys
if '-filter_complex' in sys.argv: raise SystemExit(9)
os.execv({self.ffmpeg!r}, [{self.ffmpeg!r}, *sys.argv[1:]])
''')
        wrapper.chmod(0o755)
        self.env['H3_FFMPEG'] = str(wrapper)
        report = self.run_script('#overlay-segment-name\n### a\n#size 96x96\n#frames 22\nred', 1)
        self.assertEqual(report['segments'][0]['status'], 'planned')
        self.assertFalse(list((self.out/'.render').glob('*.h3av')))
        self.assertFalse((self.out/'a.mp4').exists())

    def test_overlay_nomerge_small_canvas_long_unicode_and_punctuation_names(self):
        names = ['A: [B], % C; "D"', 'snow \u96ea', 'long name ' * 20]
        text = '#overlay-segment-name\n#nomerge\n' + ''.join(
            f'### {name}\n#size 32x32\n#frames 22\nred\n' for name in names)
        report = self.run_script(text)
        self.assertEqual(report['merged'], [])
        self.assertEqual(len(list(self.out.glob('*.mp4'))), 3)
        for row in report['segments']:
            data = self.rgb(Path(row['video']))
            self.assertEqual(len(data), 22*32*32*3)
            self.assertNotEqual(data, self.rgb(Path(row['original_video'])))

    def test_metadata_validation_avoids_decoding_fourteen_segments(self):
        self.probe_wrapper(on_scan="raise SystemExit('Unexpected full frame scan')")
        text = ''.join(f'### shot{i}\n#size {"64x128" if i%2 else "128x64"}\n#frames 362\nred\n'
                       for i in range(14))
        report = self.run_script(text)
        self.assertEqual(report['merged'][0]['frames'], 14*362)
        self.assertTrue(report['merged'][0]['lossless_normalization'])
        calls = self.probe_calls()
        self.assertTrue(any(c['name'].endswith('.mov') for c in calls))
        self.assertTrue(any(c['name']=='merge-0000.mp4' for c in calls))
        self.assertFalse(any(c['scan'] for c in calls))
        self.assertEqual(report['validation'], dict(verify_frames=False, scan_timeout_seconds=600))

    def test_missing_container_counts_fall_back_to_video_scan(self):
        self.probe_wrapper(edit='''if not scan and name in ('0000.mp4', '0001.mp4', '0002.mp4'):
    video = next(s for s in data['streams'] if s['codec_type']=='video')
    if name == '0000.mp4': video.pop('nb_frames', None)
    else: video['nb_frames'] = 'N/A' if name == '0001.mp4' else '0'
''')
        self.env['H3_RENDER_SCAN_TIMEOUT'] = '720'
        report = self.run_script('### a\n#size 96x96\n#frames 22\nred\n### b\nblue\n### c\nred\n#nomerge')
        scans = [c for c in self.probe_calls() if c['scan']]
        self.assertEqual([c['name'] for c in scans], ['0000.mp4','0001.mp4','0002.mp4'])
        for call in scans:
            self.assertEqual(call['args'][call['args'].index('-select_streams')+1], 'v:0')
        self.assertEqual(report['validation']['scan_timeout_seconds'], 720)

    def test_wrong_container_count_is_rejected_without_scan(self):
        self.probe_wrapper(edit='''if name == '0000.mp4':
    next(s for s in data['streams'] if s['codec_type']=='video')['nb_frames'] = '21'
''', on_scan="raise SystemExit('A conflicting count must not trigger a scan')")
        report = self.run_script('### a\n#size 96x96\n#frames 22\nred\n#nomerge', 1)
        self.assertIn('0000.mp4: expected', report['error'])
        self.assertFalse((self.out/'a.mp4').exists())
        self.assertFalse(any(c['scan'] for c in self.probe_calls()))

    def test_metadata_geometry_rate_and_durations_are_still_checked(self):
        for kind, key, value in (('video','width',128), ('video','avg_frame_rate','30/1'),
                                 ('video','duration','9'), ('audio','duration','9')):
            with self.subTest(kind=kind, key=key):
                self.out = self.root/(kind+'-'+key)
                self.probe_wrapper(edit=f'''if name == '0000.mp4':
    next(s for s in data['streams'] if s['codec_type']=={kind!r})[{key!r}] = {value!r}
''', on_scan="raise SystemExit('Invalid metadata must not trigger a scan')")
                report = self.run_script('### a\n#size 96x96\n#frames 22\nred\n#nomerge', 1)
                self.assertIn('0000.mp4: expected', report['error'])
                self.assertFalse((self.out/'a.mp4').exists())

    def test_explicit_frame_verification_checks_decoded_count(self):
        self.env['H3_RENDER_VERIFY_FRAMES'] = '1'
        for count in ('22', '21', 'N/A'):
            with self.subTest(count=count):
                self.out = self.root/('verified-'+count.replace('/','-'))
                self.probe_wrapper(edit=f'''if scan and name == '0000.mp4':
    next(s for s in data['streams'] if s['codec_type']=='video')['nb_read_frames'] = {count!r}
''')
                report = self.run_script('### a\n#size 96x96\n#frames 22\nred\n#nomerge', 0 if count=='22' else 1)
                self.assertTrue(report['validation']['verify_frames'])
                self.assertTrue(any(c['scan'] and c['name']=='0000.mp4' for c in self.probe_calls()))
                self.assertEqual((self.out/'a.mp4').exists(), count=='22')

    def test_failed_frame_scan_identifies_file_and_preserves_clips(self):
        self.probe_wrapper(edit='''if not scan and name == '0001.mp4':
    next(s for s in data['streams'] if s['codec_type']=='video').pop('nb_frames', None)
''', on_scan="raise SystemExit(9)")
        report = self.run_script('### a\n#size 96x96\n#frames 22\nred\n### b\nblue\n#nomerge', 1)
        self.assertIn('0001.mp4: frame scan:', report['error'])
        self.assertTrue((self.out/'a.mp4').exists())
        self.assertFalse((self.out/'b.mp4').exists())

    def test_frame_scan_timeout_stops_probe_and_preserves_clips(self):
        self.probe_wrapper(edit='''if not scan and name == '0001.mp4':
    next(s for s in data['streams'] if s['codec_type']=='video').pop('nb_frames', None)
''', on_scan='''def stop(sig, frame):
    (root/'probe-stopped').touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
time.sleep(30)
''')
        self.env['H3_RENDER_SCAN_TIMEOUT'] = '0.5'
        started = time.monotonic()
        report = self.run_script('### a\n#size 96x96\n#frames 22\nred\n### b\nblue\n#nomerge', 1)
        self.assertLess(time.monotonic()-started, 15)
        self.assertIn('0001.mp4: frame scan:', report['error'])
        self.assertIn('timed out', report['error'])
        self.assertTrue((self.root/'probe-stopped').exists())
        self.assertTrue((self.out/'a.mp4').exists())
        self.assertFalse((self.out/'b.mp4').exists())

    def test_cancellation_stops_frame_scan(self):
        self.probe_wrapper(edit='''if not scan and name == '0000.mp4':
    next(s for s in data['streams'] if s['codec_type']=='video').pop('nb_frames', None)
''', on_scan='''def stop(sig, frame):
    (root/'probe-stopped').touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
(root/'probe-ready').touch()
time.sleep(30)
''')
        process = subprocess.Popen(self.command('### a\n#size 96x96\n#frames 22\nred\n#nomerge'),
                                   env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+15
            while not (self.root/'probe-ready').exists() and process.poll() is None and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertTrue((self.root/'probe-ready').exists(), 'frame scan did not start')
            process.send_signal(signal.SIGINT)
            output, _ = process.communicate(timeout=10)
            (self.root/'driver.log').write_bytes(output)
            self.assertEqual(process.returncode, 130, output.decode())
            self.assertTrue((self.root/'probe-stopped').exists())
            report = json.loads((self.out/'run.json').read_text())
            self.assertEqual(report['status'], 'interrupted')
            self.assertFalse((self.out/'a.mp4').exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()

    def test_invalid_probe_options_fail_before_outputs(self):
        for key, value in [('H3_RENDER_SCAN_TIMEOUT', value) for value in ('0','-1','nan','inf','1e999','600seconds')] + [
                ('H3_RENDER_VERIFY_FRAMES','true'), ('H3_RENDER_VERIFY_FRAMES','2')]:
            with self.subTest(key=key, value=value):
                result = subprocess.run(self.command('### a\nred'), env=self.env|{key:value},
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(key, result.stderr)
                self.assertFalse(self.out.exists())

    def test_stream_copy_multiple_chains_and_independent_shots(self):
        report = self.run_script('### first clip\n#size 96x96\n#frames 56\nred\n'
                                 "### second 'clip\n#continue\nblue\n### independent\n#frames 22\nred\n"
                                 '!### skipped\n')
        names = ['first clip.mp4', "second 'clip.mp4", 'independent.mp4']
        original = b''.join(self.rgb(self.out/n) for n in names)
        self.assertEqual(original, self.rgb(self.out/'scenes.mp4'))
        self.assertEqual(original[:73*96*96*3], self.rgb(self.out/'first clip-merged.mp4'))
        self.assertEqual([m['frames'] for m in report['merged']], [73, 95])
        self.assertTrue(all(not m['lossless_normalization'] for m in report['merged']))
        self.assertFalse((self.out/'skipped.mp4').exists())
        pcm = array.array('h', subprocess.check_output([self.ffmpeg, '-v', 'error', '-i', str(self.out/'scenes.mp4'),
                                                       '-map', '0:a:0', '-ac', '1', '-ar', '32000', '-f', 's16le', '-']))
        if sys.byteorder != 'little':
            pcm.byteswap()
        for offset, expected in ((.25,440), (56/24+.25,660), (73/24+.25,440)):
            window = pcm[int(offset*32000):int(offset*32000)+2000]
            def power(frequency):
                phase = 2*math.pi*frequency/32000
                return sum(v*math.cos(phase*i) for i,v in enumerate(window))**2 + sum(v*math.sin(phase*i) for i,v in enumerate(window))**2
            self.assertGreater(power(expected), 10*power(660 if expected==440 else 440))

    def test_different_encodings_preserve_every_decoded_frame(self):
        report = self.run_script('### a\n#size 96x96\n#frames 56\nred\n'
                                 '### b\n#continue\n#lossless-video\nblue\n')
        original = self.rgb(self.out/'a.mp4') + self.rgb(self.out/'b.mp4')
        self.assertEqual(original, self.rgb(self.out/'a-merged.mp4'))
        self.assertEqual(original, self.rgb(self.out/'scenes.mp4'))
        self.assertTrue(all(m['lossless_normalization'] for m in report['merged']))

    def test_different_canvases_fit_and_pad(self):
        report = self.run_script('### a\n#size 96x96\n#frames 22\nred\n'
                                 '### b\n#size 128x64\nblue\n')
        expected = self.rgb(self.out/'a.mp4') + self.rgb(self.out/'b.mp4', (96,96))
        self.assertEqual(expected, self.rgb(self.out/'scenes.mp4'))
        self.assertEqual(report['merged'][0]['frames'], 44)
        self.assertTrue(report['merged'][0]['lossless_normalization'])
        self.assertEqual(len(self.rgb(self.out/'b.mp4')), 22*128*64*3)

    def test_one_clip_still_produces_script_named_file(self):
        report = self.run_script('### a\n#size 96x96\n#frames 22\nred\n')
        self.assertEqual(self.rgb(self.out/'a.mp4'), self.rgb(self.out/'scenes.mp4'))
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['merged'][0]['segments'], ['a.mp4'])

    def test_nomerge_at_end_keeps_continuation_and_expanded_prompts(self):
        report = self.run_script('### a\n#size 96x96\n#frames 56\nsummary:\nred\n'
                                 'overall_soundscape:\nSoft water.\n### b\n#continue\nsummary:\nblue\n#nomerge')
        self.assertTrue(report['nomerge'])
        self.assertEqual(report['merged'], [])
        self.assertEqual(sorted(p.name for p in self.out.glob('*.mp4')), ['a.mp4','b.mp4'])
        self.assertEqual(report['segments'][1]['prompt'], 'summary:\nblue\n\noverall_soundscape:\nSoft water.')
        self.assertEqual(report['segments'][1]['delivered'], 17)

    def test_failed_render_preserves_completed_clips_without_full_merge(self):
        report = self.run_script('### a\n#size 96x96\n#frames 22\nred\n### b\nFAIL_WORKER', 1)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual([s['status'] for s in report['segments']], ['complete', 'failed'])
        self.assertEqual(sorted(p.name for p in self.out.glob('*.mp4')), ['a.mp4'])
        self.assertNotEqual(report['merged'][0]['status'], 'complete')

    def test_failed_merge_does_not_publish_final_filename(self):
        wrapper = self.root/'fail-merge.py'
        wrapper.write_text(f'''#!{sys.executable}
import os,sys
if 'concat' in sys.argv:
    raise SystemExit(9)
os.execv({self.ffmpeg!r}, [{self.ffmpeg!r}, *sys.argv[1:]])
''')
        wrapper.chmod(0o755)
        self.env['H3_FFMPEG'] = str(wrapper)
        report = self.run_script('### a\n#size 96x96\n#frames 22\nred', 1)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['merged'][0]['status'], 'failed')
        self.assertTrue((self.out/'a.mp4').exists())
        self.assertFalse((self.out/'scenes.mp4').exists())

    def test_missing_media_fails_before_workers(self):
        self.env['H3_FFPROBE'] = str(self.root/'missing-ffprobe')
        report = self.run_script('### a\n#size 96x96\n#frames 22\nred', 1)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['segments'][0]['status'], 'planned')
        self.assertFalse((self.out/'a.mp4').exists())

    def test_existing_output_directory_is_preserved(self):
        self.out.mkdir()
        sentinel = self.out/'sentinel'
        sentinel.write_text('keep')
        result = subprocess.run(self.command('### a\nred'), env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(list(self.out.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_filesystem_unicode_aliases_are_checked_before_workers(self):
        composed, decomposed = 'é', 'e\u0301'
        probe = self.root/composed
        probe.touch()
        aliases = (self.root/decomposed).exists()
        probe.unlink()
        report = self.run_script(f'### {composed}\n#size 96x96\n#frames 22\nred\n### {decomposed}\nblue\n#nomerge', 1 if aliases else 0)
        if aliases:
            self.assertIn('collides', report['error'])
            self.assertTrue(all(row['status']=='planned' for row in report['segments']))
            self.assertEqual(list(self.out.glob('*.mp4')), [])
        else:
            self.assertEqual(len(list(self.out.glob('*.mp4'))), 2)

    def test_publication_does_not_overwrite_a_concurrent_output(self):
        report = self.run_script('### a\n#size 96x96\n#frames 22\nred RACE_OUTPUT', 1)
        self.assertEqual((self.out/'a.mp4').read_bytes(), b'concurrent output; preserve me')
        self.assertEqual(report['status'], 'failed')
        self.assertFalse((self.out/'scenes.mp4').exists())

    def test_cancellation_stops_worker_and_grandchild(self):
        child_script = self.root/'child.py'
        child_script.write_text(f'''import os,signal,time
from pathlib import Path
root=Path({str(self.root)!r})
def stop(sig,frame):
    (root/'child-stopped').touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
(root/'child-ready').write_text(str(os.getpid()))
while True: time.sleep(.05)
''')
        self.worker.write_text(f'''#!{sys.executable}
import os,signal,subprocess,sys,time
from pathlib import Path
root=Path({str(self.root)!r})
child=subprocess.Popen([sys.executable,str(root/'child.py')])
def stop(sig,frame):
    child.wait(timeout=5)
    (root/'parent-stopped').touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
while not (root/'child-ready').exists(): time.sleep(.01)
(root/'ready').write_text(str(os.getpid()))
while True: time.sleep(.05)
''')
        process = subprocess.Popen(self.command('### a\nred'), env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+15
            while not (self.root/'ready').exists() and process.poll() is None and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertTrue((self.root/'ready').exists(), 'worker did not start')
            process.send_signal(signal.SIGINT)
            output, _ = process.communicate(timeout=10)
            (self.root/'driver.log').write_bytes(output)
            self.assertEqual(process.returncode, 130, output.decode())
            for name in ('child-stopped', 'parent-stopped'):
                self.assertTrue((self.root/name).exists(), name)
            report = json.loads((self.out/'run.json').read_text())
            self.assertEqual(report['status'], 'interrupted')
            self.assertEqual(report['segments'][0]['status'], 'interrupted')
            self.assertFalse((self.out/'scenes.mp4').exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            for name in ('ready', 'child-ready'):
                if (self.root/name).exists():
                    try:
                        os.kill(int((self.root/name).read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out:
        OUTPUT = args.out.resolve()
        OUTPUT.mkdir(parents=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeMediaTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if OUTPUT:
        record = dict(passed=result.wasSuccessful(), tests=result.testsRun,
                      failures=len(result.failures), errors=len(result.errors),
                      driver_sha256=hashlib.sha256(DRIVER.read_bytes()).hexdigest())
        (OUTPUT/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
