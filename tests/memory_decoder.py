#!/usr/bin/env python3
"""Current decoder media checks, failure cleanup, and allocation accounting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command, **kw):
    return subprocess.run(list(map(str, command)), check=True, **kw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    out = ROOT/'outputs/memory-validation'/('decoder-sanitize' if args.sanitize else 'decoder')
    out.mkdir(parents=True, exist_ok=True)
    binaries = {}
    for label, source in [('current', ROOT/'src/media/ffmpeg.c')]:
        binary = ROOT/'bin'/('memory_decoder_'+('sanitize_' if args.sanitize else '')+label)
        binary.parent.mkdir(parents=True,exist_ok=True)
        command = ['clang', '-std=c11', '-D_DARWIN_C_SOURCE', '-I', ROOT, '-O1', '-g',
                   '-DH3_DECODER_SOURCE="'+str(source)+'"', ROOT/'tests/memory_decoder_probe.c',
                   ROOT/'src/memory.c', ROOT/'bin/libh3.a', '-o', binary]
        command += shlex.split(os.environ.get('H3_TEST_LDLIBS', '-framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Accelerate -framework CoreML -framework CoreVideo -licucore -lc++ -lm'))
        if args.sanitize:
            command += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        run(command)
        binaries[label] = binary
    fake = out/'fake-ffmpeg'
    fake.write_text('#!'+sys.executable+'\n'+r'''
import os, sys, time, signal
mode = sys.argv[sys.argv.index('-i')+1]
limit = int(sys.argv[sys.argv.index('-frames:v')+1])
if mode == 'failure': sys.exit(9)
if mode == 'empty': sys.exit(0)
if mode == 'stall': time.sleep(20); sys.exit(0)
if mode == 'ignore-term':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(20); sys.exit(0)
size = 7 * 3 * 3
frames = min(limit, 9)
raw = bytes((i * 37 + i//size) % 256 for i in range(frames*size))
if mode == 'partial': raw = raw[:-1]
if mode == 'extra': raw += b'x'
# Deliberately split frames into many writes, with short pipe reads.
for i in range(0, len(raw), 13):
    os.write(1, raw[i:i+13])
    time.sleep(.001)
if mode == 'completion-stall':
    os.close(1); time.sleep(20)
''')
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith('H3_')}
    records = []

    def decode(label, name, path, cap, width=48, height=32, extra=None, cancel=0):
        dest = out/(name+'-'+label+'.f32')
        result = subprocess.run([str(binaries[label]), str(path), str(width), str(height), str(cap),
            str(dest), str(cancel)], capture_output=True, text=True,
            env=env | (extra or {}), timeout=60)
        assert result.returncode == 0, (name, label, result.returncode, result.stderr)
        data = json.loads(result.stdout)
        data.update(name=name, version=label, error=result.stderr.strip())
        if data['ok']:
            data['sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
            assert dest.stat().st_size == data['bytes']
        records.append(data)
        return data

    # Real variable colors, motion, dimensions and FPS, including long clips.
    for frames, fps in [(1, 24), (9, 24), (90, 30), (360, 24)]:
        source = out/f'video-{frames}-{fps}.mkv'
        run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
             f'testsrc=size=64x48:rate={fps}', '-frames:v', frames, '-c:v', 'ffv1', source])
        normalized = frames*24//fps
        caps = sorted({1, 5, max(5, normalized-1), normalized, normalized+7})
        for cap in caps:
            name = f'real-{frames}-{fps}-cap{cap}'
            data = decode('current', name, source, cap)
            assert data['ok'] and data['frames'] == min(cap, normalized), data
            assert data['first'] == cap*48*32*3*4
            assert data['second'] == 48*32*3
            assert data['peak'] == data['first']+data['second']
    for mode, accepted in [('short', 1), ('partial', 0), ('empty', 0), ('failure', 0)]:
        data = decode('current', mode, mode, 15, 7, 3, {'H3_FFMPEG': str(fake)})
        assert data['ok'] == accepted, data
        if accepted:
            import struct
            raw = bytes((i * 37 + i//63) % 256 for i in range(9*63))
            actual = struct.unpack('<'+'f'*len(raw), (out/(mode+'-current.f32')).read_bytes())
            assert all(abs(x-y/255.) < 1e-7 for x,y in zip(actual,(raw[t*63+pixel*3+c] for c in range(3) for t in range(9) for pixel in range(21))))
    for allocation in (1, 2):
        data = decode('current', f'oom{allocation}', 'short', 15, 7, 3,
            {'H3_TEST_FAIL_ALLOCATION_AT': str(allocation), 'H3_FFMPEG': str(fake)})
        assert not data['ok'] and 'out of memory' in data['error']
    data = decode('current', 'extra-byte', 'extra', 9, 7, 3, {'H3_FFMPEG': str(fake)})
    assert not data['ok'] and 'bounded video' in data['error']
    data = decode('current', 'read-error', 'short', 15, 7, 3,
        {'H3_FFMPEG': str(fake), 'H3_TEST_READ_FAILURE_AT': '4'})
    assert not data['ok'] and 'cannot read FFmpeg' in data['error']
    for width, height, cap, message in [
        (1500000000, 1000000000, 1, 'reserve overflows'),
        (2147483647, 2147483647, 1, 'converted video size overflows'),
        (2147483647, 2147483647, 2147483647, 'decoded video size overflows')]:
        data = decode('current', 'overflow-'+message.split()[0], 'unused', cap,
            width, height)
        assert not data['ok'] and data['calls'] == 0 and message in data['error']
    for path in (out/'does-not-exist.mp4', out/'malformed.mp4'):
        if path.name == 'malformed.mp4': path.write_bytes(b'invalid video')
        data = decode('current', path.stem, path, 15)
        assert not data['ok']
    data = decode('current', 'spawn-failure', 'unused', 15, extra={'H3_FFMPEG': '/missing-ffmpeg'})
    assert not data['ok'] and 'cannot start' in data['error']
    for query in (1, 8):
        data = decode('current', f'memory-cancel{query}', 'short', 15, 7, 3,
            {'H3_FFMPEG': str(fake)}, cancel=query)
        assert not data['ok'] and 'reclaimable physical memory' in data['error']
    data = decode('current', 'stalled-child-cancel', 'stall', 15, 7, 3,
        {'H3_FFMPEG': str(fake)}, cancel=5)
    assert not data['ok'] and 'reclaimable physical memory' in data['error']
    for mode, cap, query in [('ignore-term', 15, 5), ('completion-stall', 9, 100)]:
        data = decode('current', mode, mode, cap, 7, 3, {'H3_FFMPEG': str(fake)}, cancel=query)
        assert not data['ok'] and 'reclaimable physical memory' in data['error']
    (out/'results.json').write_text(json.dumps(records, indent=2)+'\n')
    print(f'ok: {len(records)} decoder runs; normalized F32 media, one-frame staging, errors, allocation failures, cancellation and child cleanup')


if __name__ == '__main__': main()
