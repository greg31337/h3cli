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
import math, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
def value(name, default=None):
    return args[args.index('--'+name)+1] if '--'+name in args else default
prompt = value('prompt', '')
if 'RACE_OUTPUT' in prompt:
    (Path(value('output')).parent.parent/'a.mp4').write_bytes(b'concurrent output; preserve me')
if 'FAIL_WORKER' in prompt:
    print('synthetic worker failure', file=sys.stderr)
    raise SystemExit(7)
frames = int(value('frames', '56'))
if '--seconds' in args:
    frames = math.floor(float(value('seconds'))*24+.5)
frames = max(frames, 5) + (5-max(frames, 5))%17
if '--continue-from' in args:
    assert Path(value('continue-from')).is_file()
    frames -= int(value('continue-context', '39'))
width, height = int(value('width','864')), int(value('height','480'))
color = 'blue' if 'blue' in prompt else 'red'
frequency = 660 if color == 'blue' else 440
encoding = ['-c:v', 'libx264rgb', '-crf', '0', '-pix_fmt', 'rgb24'] if '--lossless-video' in args else ['-c:v','libx264','-crf','18','-pix_fmt','yuv420p']
command = [os.environ['H3_FFMPEG'], '-nostdin','-v','error','-n', '-f','lavfi','-i',
           f'color=c={{color}}:s={{width}}x{{height}}:r=24:d={{frames/24:.12f}}',
           '-f','lavfi','-i',f'sine=frequency={{frequency}}:sample_rate=32000','-t',f'{{frames/24:.12f}}',
           '-frames:v',str(frames),*encoding,'-c:a','aac','-ar','32000','-ac','2',value('output')]
subprocess.run(command, check=True)
state = Path(value('save-av-state'))
state.write_text('Synthetic test state; not inference evidence.\\n')
Path(str(state)+'.presentation').write_text('Synthetic presentation.\\n')
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
