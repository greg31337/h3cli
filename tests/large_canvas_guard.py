"""Serial CUDA campaign launcher; never accepts an arbitrary inference command."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

GIB = 1 << 30


def plan(kind, width, height, frames, *, rows=None, refine=4):
    if kind not in ('native', 'upscale', 'continuation'):
        raise ValueError('unknown job kind')
    if frames not in (22, 39) and not (kind == 'continuation' and frames == 56):
        raise ValueError('only 22/39 frames; continuation alone permits 56')
    if min(width, height) < 32 or width % 32 or height % 32:
        raise ValueError('unaligned canvas')
    axis, area = (3840, 8355840) if kind == 'upscale' else (1920, 2088960)
    if max(width, height) > axis or width * height > area:
        raise ValueError('canvas exceeds operation limit')
    if kind == 'continuation' and (max(width, height) > 1344 or min(width, height) > 768):
        raise ValueError('continuation is limited to 1344x768')
    if kind == 'upscale' and (width % 64 or height % 64 or refine not in (0, 2, 3, 4)):
        raise ValueError('upscale must double an aligned source and use K=0/2/3/4')
    t = 2 + 5 * ((frames - 5) // 17)
    a = (frames * 40 + 12) // 24
    minimum = t * (width // 32) * (height // 32) + 2 * a
    sequence = minimum + 512 if rows is None else rows
    if sequence < minimum or sequence > 1000000:
        raise ValueError('unsupported sequence element indexing')
    future = sequence * 358400 + GIB
    slots = 1541406720
    live = max(7 * GIB, future + slots + 2 * GIB)
    if kind == 'upscale' and refine == 0:
        live = max(7 * GIB, t * (width // 16) * (height // 16) * 512 * 2 * 4 + 2 * GIB)
    capacity = math.ceil(live / .9)
    if capacity > 60 * GIB:
        raise ValueError('development capacity estimate exceeds 60 GiB')
    return dict(kind=kind, width=width, height=height, frames=frames,
                sequence_rows=sequence, live_bytes=live, capacity_bytes=capacity)


def process_tree(root):
    """Include workers that create their own process group/session."""
    processes = {}
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            fields = (entry/'stat').read_text().rsplit(')',1)[1].split()
            match = re.search(r'VmRSS:\s+(\d+)', (entry/'status').read_text())
            processes[int(entry.name)] = (int(fields[1]), int(match[1]) if match else 0)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    children = {root}
    while True:
        found = {pid for pid,(parent,_) in processes.items() if parent in children}
        if found <= children:
            break
        children |= found
    return children, sum(processes[pid][1] for pid in children if pid in processes)


def select_frames(kind,width,height,reference_rows=0,refine=4):
    """Select before creating any source; report whether 39 needed to fall back."""
    for frames in (39,22):
        base=plan(kind,width,height,frames,refine=refine)
        try:
            return plan(kind,width,height,frames,rows=base['sequence_rows']+reference_rows,refine=refine)
        except ValueError as error:
            if 'capacity' not in str(error) or frames==22:
                raise


def device_admission(p,total_mib,free_mib):
    reserve=max(GIB,total_mib*(1<<20)//10)
    required=p['live_bytes']+reserve
    if p['capacity_bytes']>60*GIB or required>60*GIB:
        raise ValueError('development live memory plus physical-device reserve exceeds 60 GiB')
    if free_mib*(1<<20)<required:
        raise ValueError('insufficient free VRAM including physical-device reserve')
    return reserve


def run_case(command, p, output, *, media=True, env=None):
    """Caller holds campaign lock; commands and inspected inputs come from a fixed runner."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    record = output.with_suffix('.run.json')
    if output.exists() or record.exists():
        raise ValueError('case/output already exists')
    total, free = map(int, subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.free',
        '--format=csv,noheader,nounits'], text=True).strip().split(','))
    reserve=device_admission(p,total,free)
    available = int(re.search(r'MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text())[1]) * 1024
    if available < 32 * GIB:
        raise ValueError('insufficient available host RAM for this campaign')
    env = dict(env or os.environ, H3_CUDA_WEIGHT_MODE='stream', H3_PROFILE='1', H3_TEST_MAX_EVALUATIONS='6', H3_OFFLINE='1')
    started = time.monotonic()
    metadata = dict(plan=p, command=command, gpu_total_mib=total, gpu_free_mib=free,
                    device_reserve_bytes=reserve,host_available_bytes=available, status='running')
    record.write_text(json.dumps(metadata, indent=2) + '\n')
    peak_used_mib, peak_rss_kib = total-free, 0
    with output.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                used = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'], text=True).strip())
                peak_used_mib = max(peak_used_mib, used)
                _, rss = process_tree(process.pid)
                peak_rss_kib = max(peak_rss_kib, rss)
                if used * (1 << 20) > 60 * GIB or time.monotonic()-started > 3600:
                    raise RuntimeError('device usage or one-hour deadline exceeded')
                time.sleep(.5)
            code = process.wait(timeout=60)
            if code:
                raise RuntimeError(f'case exited {code}')
            if media:
                info = json.loads(subprocess.check_output([env.get('H3_FFPROBE', 'ffprobe'), '-v', 'error',
                    '-count_frames', '-show_streams', '-of', 'json', str(output)], text=True))
                video, = [s for s in info['streams'] if s['codec_type'] == 'video']
                assert (video['width'], video['height'], int(video['nb_read_frames'])) == (p['width'], p['height'], p.get('delivered', p['frames']))
                assert video['avg_frame_rate'] == '24/1' and float(video['duration']) <= 3
                metadata['media'] = info
            metadata['status'] = 'passed'
        except BaseException as error:
            metadata.update(status='failed', guard_failure=str(error))
            if process.poll() is None:
                descendants,_ = process_tree(process.pid)
                for child in descendants-{process.pid}:
                    try:
                        os.kill(child, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            raise
        finally:
            metadata.update(exit_code=process.returncode, seconds=time.monotonic()-started,
                peak_device_used_mib=peak_used_mib, peak_process_rss_kib=peak_rss_kib,
                rss_scope='sum of resident descendant pages; shared pages may be counted twice')
            record.write_text(json.dumps(metadata, indent=2) + '\n')
    record.write_text(json.dumps(metadata, indent=2) + '\n')
    if metadata['status'] != 'passed':
        raise SystemExit(code or 1)
    return metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('kind', choices=('native', 'upscale'))
    ap.add_argument('--binary', default='bin/h3cli')
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1088)
    ap.add_argument('--frames', type=int, default=22)
    ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--refine', type=int, default=4)
    ap.add_argument('--source')
    ap.add_argument('--upscale-model')
    ap.add_argument('--prompt', default='A polished red ceramic teapot on a pale wooden table, soft studio lighting, slow camera movement.')
    args = ap.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('output already exists; use a fresh case path')
    record = output.with_suffix('.run.json')
    if record.exists():
        raise ValueError('case record already exists')
    binary = str(Path(args.binary).resolve())
    command = [binary, '-d', args.model, '--verbose']
    with open('/tmp/h3-large-canvas-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.kind == 'upscale':
            if not args.source:
                raise ValueError('upscale needs an owned .h3up source')
            info = subprocess.check_output([binary, '--inspect-upscale-state', args.source], text=True)
            match = re.search(r'source=(\d+)x(\d+) target=(\d+)x(\d+) frames=(\d+).*? rows=(\d+)', info)
            if not match:
                raise ValueError('cannot inspect upscale source')
            sw, sh, w, h, frames, rows = map(int, match.groups())
            if (w, h) != (2 * sw, 2 * sh):
                raise ValueError('source/target geometry mismatch')
            if frames != args.frames:
                raise ValueError('source duration differs from declared case')
            p = plan('upscale', w, h, frames, rows=rows, refine=args.refine)
            command += ['--upscale-state', args.source, '--upscale-refine-steps', str(args.refine)]
            if args.upscale_model:
                command += ['--upscale-model', args.upscale_model]
        else:
            p = plan('native', args.width, args.height, args.frames)
            if not 1 <= args.steps <= 1000:
                raise ValueError('invalid steps')
            command += ['-p', args.prompt, '--width', str(args.width), '--height', str(args.height),
                        '--frames', str(args.frames), '--steps', str(args.steps), '--seed', '42',
                        '--save-upscale-state', str(output.with_suffix('.h3up'))]
        command += ['--save-av-state', str(output.with_suffix('.h3av')), '-o', str(output)]
        run_case(command, p, output)


if __name__ == '__main__':
    main()
