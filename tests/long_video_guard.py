"""256x256 / under-20-second campaign policy and serial CUDA monitor.

Only the fixed long_video_campaign entry point launches model work.
"""
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


def plan(kind, width, height, frames, *, context=0, merged=None, steps=2, cuda=True, reference_frames=0):
    if kind not in ('native', 'continuation', 'decode', 'script', 'http', 'api'):
        raise ValueError('unsupported bounded campaign operation')
    if (width,height)!=(256,256):
        raise ValueError('every runtime canvas must be 256x256')
    if frames<22 or frames>464 or (frames-5)%17 or (not cuda and frames>362):
        raise ValueError('aligned frames must fit the backend and 20-second envelope')
    if steps<1 or steps>6:
        raise ValueError('at most six full-depth evaluations')
    if reference_frames<0 or reference_frames>360:
        raise ValueError('reference input exceeds its independent 15-second cap')
    if context and (context<39 or (context-39)%51 or context>=frames):
        raise ValueError('invalid preserved context')
    delivered=frames-context
    if merged is not None and (merged<delivered or merged>480):
        raise ValueError('merged delivery exceeds 20 seconds')
    t=2+5*((frames-5)//17); a=(frames*40+12)//24
    # Include an intentionally generous reference/text reserve in admission.
    sequence=t*64+2*a+8192
    live=max(7*GIB, sequence*358400+GIB+1541406720+2*GIB)
    capacity=math.ceil(live/.9)
    if capacity>60*GIB: raise ValueError('development capacity exceeds 60 GiB')
    return dict(kind=kind,width=width,height=height,frames=frames,delivered=delivered,
        sequence_rows=sequence,live_bytes=live,capacity_bytes=capacity)


def inspect_video(path, *, max_frames=480, env=None):
    env=env or os.environ
    info=json.loads(subprocess.check_output([env.get('H3_FFPROBE','ffprobe'),'-v','error',
        '-count_frames','-show_streams','-of','json',str(path)],text=True))
    video,=[s for s in info['streams'] if s['codec_type']=='video']
    if (video['width'],video['height'])!=(256,256) or int(video['nb_read_frames'])>max_frames or float(video['duration'])>min(20,max_frames/24) or video['avg_frame_rate']!='24/1':
        raise ValueError('media exceeds bounded development envelope')
    for stream in info['streams']:
        if stream['codec_type']=='audio' and abs(float(stream['duration'])-float(video['duration']))>0.1:
            raise ValueError('audio/video timestamps diverge')
    return info


def inspect_state(path):
    import struct
    with Path(path).open('rb') as f: header=f.read(160)
    if header[:8]!=b'H3AV\r\n\x1a\n': raise ValueError('expected AV state')
    w,h,frames=struct.unpack_from('<III',header,24)
    return plan('decode',w,h,frames)

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
                assert video['avg_frame_rate'] == '24/1' and float(video['duration']) <= 20
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

