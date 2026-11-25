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


def plan(kind, width, height, frames, *, context=0, merged=None, steps=2, cuda=True, reference_frames=0,
         precision='bf16', placement='stream', attention='default'):
    if kind not in ('native', 'continuation', 'decode', 'script', 'http', 'api'):
        raise ValueError('unsupported bounded campaign operation')
    if (width,height)!=(256,256):
        raise ValueError('every runtime canvas must be 256x256')
    if frames<22 or frames>464 or (frames-5)%17 or (not cuda and frames>362):
        raise ValueError('aligned frames must fit the backend and 20-second envelope')
    if steps<1 or steps>6:
        raise ValueError('at most six full-depth evaluations')
    if precision not in ('bf16','nvfp4') or placement not in ('auto','resident','stream'):
        raise ValueError('unsupported campaign precision/placement')
    if precision!='bf16' and not cuda:
        raise ValueError('NVFP4 requires CUDA')
    if attention not in ('default','sage3') or (attention!='default' and not cuda):
        raise ValueError('unsupported bounded attention/backend')
    if reference_frames<0 or reference_frames>360:
        raise ValueError('reference input exceeds its independent 15-second cap')
    if context and (context<39 or (context-39)%51 or context>=frames):
        raise ValueError('invalid preserved context')
    delivered=frames-context
    if merged is not None and (merged<delivered or merged>464):
        raise ValueError('merged delivery exceeds 20 seconds')
    t=2+5*((frames-5)//17); a=(frames*40+12)//24
    # Include an intentionally generous reference/text reserve in admission.
    sequence=t*64+2*a+8192
    # Keep reserves conservative: bounded quant scratch/cold staging <=2 GiB;
    # resident packed core <11 GiB; streamed packed operand <0.1 GiB.
    weights=(11*GIB if placement!='stream' else GIB//10) if precision=='nvfp4' else (36*GIB if placement!='stream' else 1541406720)
    attention_bytes=GIB//2 if attention=='sage3' else 0
    live=max(7*GIB, sequence*358400+GIB+weights+2*GIB+attention_bytes)
    capacity=math.ceil(live/.9)
    if capacity>60*GIB: raise ValueError('development capacity exceeds 60 GiB')
    return dict(kind=kind,width=width,height=height,frames=frames,delivered=delivered,
        source_canvas=[width,height],target_canvas=[width,height],context=context,merged=merged,
        steps=steps,precision=precision,placement=placement,attention=attention,
        attention_workspace_bytes=attention_bytes,reference_frames=reference_frames,
        sequence_rows=sequence,live_bytes=live,capacity_bytes=capacity)


def inspect_video(path, *, max_frames=464, env=None):
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
    precision,attention='bf16','default'
    sidecar=Path(str(path)+'.presentation')
    if sidecar.exists():
        if sidecar.stat().st_size>65536:raise ValueError('oversized presentation metadata')
        policy=dict(line.split(' ',1) for line in sidecar.read_text().splitlines())
        quant=int(policy.get('denoise_quant','0').split()[0]);mode=int(policy.get('attention','0').split()[0])
        if quant not in (0,2) or mode not in (0,2):raise ValueError('unsupported saved execution policy')
        precision='nvfp4' if quant==2 else 'bf16'
        attention='sage3' if mode==2 else 'default'
    return plan('decode',w,h,frames,precision=precision,attention=attention)


def inspect_checkpoint(path):
    """Read only the small identity/table, before a native loader allocates data."""
    import struct
    with Path(path).open('rb') as f:
        header=f.read(96)
        if header[:8] not in (b'H3SAMPLE',b'H3UPSRC\x01') or len(header)!=96:
            raise ValueError('expected sampler/source container')
        count=struct.unpack_from('<I',header,20)[0]
        if not 1<=count<=64:raise ValueError('invalid bounded section table')
        entries=f.read(72*count)
        if len(entries)!=72*count:raise ValueError('truncated section table')
        precision,attention='bf16','default'
        for i in range(count):
            kind,version,flags,dtype,items,offset,size=struct.unpack_from('<4I3Q',entries,i*72)
            if kind not in (34,35):continue
            expected=8 if kind==34 else 12
            if version!=1 or flags!=1 or size!=expected or offset<96+72*count:
                raise ValueError('invalid bounded execution policy')
            f.seek(offset);policy=f.read(size)
            if len(policy)!=size:raise ValueError('truncated execution policy')
            if kind==34:
                mode,recipe=struct.unpack('<2I',policy)
                if mode!=2 or recipe not in (2,5):raise ValueError('unsupported saved projection policy')
                precision='nvfp4'
            else:
                if struct.unpack('<3I',policy)!=(2,1,1):raise ValueError('unsupported saved attention policy')
                attention='sage3'
        for i in range(count):
            kind,version,flags,dtype,items,offset,size=struct.unpack_from('<4I3Q',entries,i*72)
            if kind==1:
                if version!=1 or size<172 or size>512 or offset<96+72*count:
                    raise ValueError('invalid bounded identity')
                f.seek(offset);identity=f.read(size)
                if len(identity)!=size:raise ValueError('truncated identity')
                w,h,frames=struct.unpack_from('<iii',identity,160)
                width,height=struct.unpack_from('<ii',identity,16)
                steps=struct.unpack_from('<i',identity,4)[0]
                if (width,height)!=(w,h):raise ValueError('hidden render canvas')
                return plan('decode',w,h,frames,steps=steps,precision=precision,attention=attention)
    raise ValueError('missing checkpoint identity')

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
    if (env or os.environ).get('H3_TEST_DIT_BLOCKS') not in (None,'50'):
        raise ValueError('bounded campaign requires all 50 DiT blocks')
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
    env = dict(env or os.environ, H3_CUDA_WEIGHT_MODE=p.get('placement','stream'), H3_PROFILE='1',
               H3_TEST_CUDA_CAPACITY_BYTES=str(60*GIB), H3_TEST_MAX_EVALUATIONS='6', H3_OFFLINE='1')
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

