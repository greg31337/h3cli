#!/usr/bin/env python3
"""Bounded, serial multi-reference CUDA campaign; no production-code changes.

prepare creates an immutable manifest; run records whole-CLI timing and NVML
telemetry. report is a separate, GPU-free command. Run from the repository root.
"""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path('outputs/multi-reference-cuda')
GIB = 1 << 30
PROTOCOL = 1
PROMPT = ('A cinematic wide shot in a sunlit courtyard. Adults wearing everyday '
          'clothing turn toward the camera and gesture naturally. The camera '
          'moves slowly, with coherent lighting and quiet outdoor ambience.')
IMAGE_NAMES = ['1.jpg', 'face1.jpg', 'face2.jpg', 'body1.jpg', 'body2.jpg',
               '2.jpg', '3.jpg', '5.png', '4.jpg']
IMAGE_ROLES = ['beach portrait', 'dark-haired outdoor portrait', 'garden portrait',
               'indoor standing portrait', 'red floral dress outdoors',
               'forest portrait', 'shoreline portrait', 'portrait with glasses',
               'white shirt portrait']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2) + '\n')
    temporary.replace(path)


def probe(path, count=False):
    cmd = ['ffprobe', '-v', 'error'] + (['-count_frames'] if count else [])
    return json.loads(subprocess.check_output(cmd + [
        '-show_format', '-show_streams', '-of', 'json', str(path)]))


def aligned(frames):
    n = max(5, frames)
    return n + (5 - n) % 17


def canvas(w, h, target_w, target_h, mode):
    scale = min(1., 2048 / min(w, h)) if mode == 'max' else min(
        1., math.sqrt(target_w * target_h / (w * h)))
    return [max(32, round(w * scale / 32) * 32),
            max(32, round(h * scale / 32) * 32)]


def cg_accounting(limit, used, stats):
    # Inactive clean file cache can be reclaimed under the cgroup limit.
    # Hashing the checkpoints can fill memory.usage without consuming the
    # nonreclaimable working set. Retain raw usage alongside this estimate.
    prefix = 'total_' if 'total_inactive_file' in stats else ''
    inactive = stats.get(prefix+'inactive_file', 0)
    dirty = stats.get(prefix+'dirty', stats.get('file_dirty', 0))
    writeback = stats.get(prefix+'writeback', stats.get('file_writeback', 0))
    reclaimable = min(used, max(0, inactive-dirty-writeback))
    available = min(limit, max(0,limit-used)+reclaimable) if limit else None
    return {'limit':limit,'used':used,'reclaimable_inactive_file':reclaimable,
            'working_set':used-reclaimable,'available_estimate':available}


def cg_memory():
    for base, limit, used in [('/sys/fs/cgroup', 'memory.max', 'memory.current'),
                              ('/sys/fs/cgroup/memory', 'memory.limit_in_bytes', 'memory.usage_in_bytes')]:
        try:
            a, b = (Path(base) / limit).read_text().strip(), (Path(base) / used).read_text().strip()
            stats=dict(line.split() for line in (Path(base)/'memory.stat').read_text().splitlines())
            return cg_accounting(int(a) if a != 'max' else None, int(b),
                                 {k:int(v) for k,v in stats.items()})
        except (OSError, ValueError):
            pass
    return {'limit': None, 'used': None, 'available_estimate': None}


def validate_case(c, assets):
    # Six-step definitions remain readable in the original frozen contract;
    # the final-run overlay requires ten steps before any main render launches.
    sampled=c['phase']=='sample'
    assert c['steps'] in ((1,2) if sampled else (2,6,10)) and c['frames'] == aligned(c['frames'])
    assert (c['width'], c['height']) in ((640, 480), (1344, 768))
    refs = [assets[k] for k in c['references']]
    images = [a for a in refs if a['kind'] == 'image']
    videos = [a for a in refs if a['kind'] == 'video']
    if c.get('scope') == 'long-362':
        assert c['frames'] == 362 and (sampled or (c['steps'] in (6,10) and c['phase']=='main'))
        assert not videos and (len(images) == 9 or (not refs and c.get('baseline')))
    else:
        assert c['frames'] <= 226
    assert len(images) <= 9 and len(videos) <= 3 and len(refs) <= 12
    assert c['image_size'] in ('match', 'max')
    assert sum(a['frames'] for a in videos) <= 144
    assert sum(a['duration'] for a in videos) <= 6.000001
    for a in videos:
        assert a['frames'] >= 48 and a['fps'] == '24/1'
        assert a['duration'] >= 2. and a['frames'] <= c['frames']
    if videos:
        assert c['frames'] >= aligned(sum(a['frames'] for a in videos))
    assert len({assets[k]['sha256'] for k in c['references']}) == len(refs)


def make_cases():
    cases = []
    for w, h in [(640, 480), (1344, 768)]:
        def case(name, images=0, videos=0, size='match', frames=None, phase='main', **extra):
            refs = [f'I{i:02d}' for i in range(1, images + 1)]
            refs += {0: [], 1: ['V1a'], 2: ['V2a', 'V2b'], 3: ['V3a', 'V3b', 'V3c']}[videos]
            c = {'id': f'R{w}-{name}', 'name': name, 'width': w, 'height': h,
                 'frames': frames or (226 if not videos else 73 if videos == 1 else 158),
                 'steps': 2 if phase == 'smoke' else 6, 'seed': 42, 'phase': phase,
                 'references': refs, 'image_size': size, 'prompt': PROMPT,
                 'bound_prompt': False, 'embedded_audio': False, **extra}
            c['baseline_key'] = f'R{w}-{phase}-F{c["frames"]:03d}'
            return c
        # Smoke includes the maximal mixed request at its full input duration.
        cases += [case('SMOKE-B00-F073', frames=73, phase='smoke', baseline=True),
                  case('SMOKE-I09-max', 9, size='max', frames=73, phase='smoke'),
                  case('SMOKE-V01', videos=1, phase='smoke'),
                  case('SMOKE-B00-F158', frames=158, phase='smoke', baseline=True),
                  case('SMOKE-X93-max', 9, 3, 'max', phase='smoke')]
        groups = {73: [], 158: [], 226: []}
        for n in (1, 3, 6, 9):
            for size in ('match', 'max'):
                groups[226].append(case(f'I{n:02d}-{size}', n, size=size))
        for n in (1, 2, 3):
            c = case(f'V{n:02d}', videos=n)
            groups[c['frames']].append(c)
        for ni, nv in [(1, 1), (9, 1), (3, 2), (9, 3)]:
            for size in ('match', 'max'):
                c = case(f'X{ni}{nv}-{size}', ni, nv, size)
                groups[c['frames']].append(c)
        for ni in (0, 1):
            c = case(f'A{ni}1', ni, 1, embedded_audio=True)
            c['references'][-1] = 'A1'
            groups[73].append(c)
        c = case('O03', 3, bound_prompt=True)
        c['references'].reverse()
        c['prompt'] += ' Use <Picture 1>, <Picture 2> and <Picture 3> for three distinct clothed people, from left to right.'
        groups[226].append(c)
        c = case('O91', 9, 1, bound_prompt=True)
        c['references'].insert(4, c['references'].pop())
        c['prompt'] += ' Use the appearances in <Picture 1> and <Picture 9> with the movement in <Video 1>, keeping everyone clothed.'
        groups[73].append(c)
        groups[73].append(case('R91-max', 9, 1, 'max', repeat_of=f'R{w}-X91-max'))
        for frames in (226, 73, 158):
            group = groups[frames]
            mid = (len(group) + 1) // 2
            cases.append(case(f'B00-F{frames:03d}-r1', frames=frames, baseline=True))
            cases += group[:mid]
            cases.append(case(f'B00-F{frames:03d}-r2', frames=frames, baseline=True))
            cases += group[mid:]
            cases.append(case(f'B00-F{frames:03d}-r3', frames=frames, baseline=True))
    assert len(cases) == 76 and len({c['id'] for c in cases}) == 76
    # All smoke checks precede main work, at lower resolution first.
    return [c for c in cases if c['phase'] == 'smoke'] + [c for c in cases if c['phase'] == 'main']


def long_cases():
    """Explicit user-authorized 362-frame extension, with its own controls."""
    cases=[]
    for w,h in ((640,480),(1344,768)):
        for name,size,baseline in [('B00-F362-r1','match',True),
                                    ('I09-F362-match','match',False),
                                    ('B00-F362-r2','match',True),
                                    ('I09-F362-max','max',False),
                                    ('B00-F362-r3','match',True)]:
            c={'id':f'R{w}-{name}','name':name,'width':w,'height':h,'frames':362,
               'steps':6,'seed':42,'phase':'main','scope':'long-362',
               'references':[] if baseline else [f'I{i:02d}' for i in range(1,10)],
               'image_size':size,'prompt':PROMPT,'bound_prompt':False,
               'embedded_audio':False,'baseline_key':f'R{w}-main-F362'}
            if baseline:c['baseline']=True
            cases.append(c)
    return cases


def load_manifest():
    m=json.loads((ROOT/'manifest.json').read_text())
    assert digest({k:v for k,v in m.items() if k!='identity'})==m['identity'],'Base manifest changed'
    path=ROOT/'extension-362.json'
    if path.exists():
        extension=json.loads(path.read_text())
        assert extension['base_identity']==m['identity']
        assert digest({k:v for k,v in extension.items() if k!='identity'})==extension['identity']
        assert extension['cases']==long_cases(),'Unexpected extension configuration'
        m['cases']+=extension['cases']
        m['extension_identity']=extension['identity']
    path=ROOT/'final-main.json'
    if path.exists():
        final=json.loads(path.read_text())
        assert final['base_identity']==m['identity']
        assert final['extension_identity']==m.get('extension_identity')
        assert digest({k:v for k,v in final.items() if k!='identity'})==final['identity']
        assert final['cases']==final_cases(m['cases']),'Final matrix changed'
        m['cases']=[c for c in m['cases'] if c['phase']=='smoke']+final['cases']
        m['final_identity']=final['identity']
        m['settings']['max_evaluations']=10
    path=ROOT/'sample-one-hour.json'
    if path.exists():
        sampled=json.loads(path.read_text())
        assert sampled['base_identity']==m['identity'] and sampled['final_identity']==m['final_identity']
        assert digest({k:v for k,v in sampled.items() if k!='identity'})==sampled['identity']
        assert sampled['cases']==sample_cases(),'Sample configuration changed'
        for c in sampled['cases']:validate_case(c,m['assets'])
        m['cases']+=sampled['cases']
        m['sampling_identity']=sampled['identity']
    return m


def sample_cases():
    cases=[]
    for width in (1344,640):
        for original in ('B00-F362-r1','I09-F362-max'):
            c=dict(next(c for c in long_cases() if c['width']==width and c['name']==original))
            name='B00-F362' if c.get('baseline') else 'I09-F362-max'
            c.update(id=f'R{width}-SAMPLE-{name}-S1',name=f'SAMPLE-{name}-S1',
                     phase='sample',steps=1,baseline_key=f'R{width}-sample-S1-F362')
            cases.append(c)
    return cases


def final_cases(cases):
    return [dict(c,steps=10,baseline_key=f'R{c["width"]}-main-S10-F{c["frames"]:03d}')
            for c in cases if c['phase']=='main']


def freeze_final():
    path=ROOT/'final-main.json'
    assert not path.exists(),'Final ten-step matrix already frozen'
    m=load_manifest()
    assert not any(r['case']['phase']=='main' for r in records()),'Main evidence already exists'
    cases=final_cases(m['cases'])
    for c in cases:validate_case(c,m['assets'])
    final={'base_identity':m['identity'],'extension_identity':m.get('extension_identity'),
        'authorization':'User requested complete final videos for every test case with --steps 10; supersedes the six-step main limit',
        'cases':cases}
    final['identity']=digest(final)
    save(path,final)
    print(f'Frozen {len(cases)} ten-step final attempts; existing smoke records remain unchanged.',flush=True)


def extend():
    base=json.loads((ROOT/'manifest.json').read_text())
    path=ROOT/'extension-362.json'
    assert not path.exists(),'Extension already frozen'
    cases=long_cases()
    for c in cases:validate_case(c,base['assets'])
    extension={'base_identity':base['identity'],
        'authorization':'User requested nine images and 362 frames at both resolutions; both image sizing modes and matching controls',
        'cases':cases}
    extension['identity']=digest(extension)
    save(path,extension)
    print('Added ten frozen long-output attempts without changing existing cases or their identity.',flush=True)


def prepare(args):
    from PIL import Image, ImageDraw
    import pynvml as nv
    assert not (ROOT / 'manifest.json').exists(), 'Refusing to overwrite frozen manifest'
    ROOT.mkdir(parents=True, exist_ok=True)
    fixtures = ROOT / 'fixtures'
    fixtures.mkdir(exist_ok=True)
    assets = {}
    for i, (name, role) in enumerate(zip(IMAGE_NAMES, IMAGE_ROLES), 1):
        source = Path('inputs') / name
        target = fixtures / name
        shutil.copy2(source, target)
        with Image.open(source) as im:
            im.load()
            w, h = im.size
            thumb = im.convert('RGB')
            thumb.thumbnail((320, 320))
            thumb.save(fixtures / f'I{i:02d}.jpg', quality=85)
        assets[f'I{i:02d}'] = {'id': f'I{i:02d}', 'kind': 'image', 'role': role,
            'path': str(target), 'original_path': str(source), 'sha256': sha(target),
            'width': w, 'height': h, 'thumbnail': f'fixtures/I{i:02d}.jpg',
            'canvases': {f'{tw}x{th}-{m}': canvas(w,h,tw,th,m)
                         for tw,th in [(640,480),(1344,768)] for m in ('match','max')}}
    assert len({a['sha256'] for a in assets.values()}) == 9
    sources = ROOT / 'fixture-sources'
    definitions = [('V1a','motion-a',72),('V2a','motion-a',72),('V2b','motion-b',72),
                   ('V3a','motion-a',48),('V3b','motion-b',48),('V3c','motion-c',48)]
    for key, source, frames in definitions:
        path = fixtures / f'{key}.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(sources/(source+'.mp4')),
            '-an','-vf','fps=24','-frames:v',str(frames),'-c:v','libx264','-preset','fast',
            '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],check=True)
        assets[key] = {'id': key, 'kind': 'video', 'path': str(path),
                       'source': source, 'expected_frames': frames, 'audio': False}
    path = fixtures / 'A1.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(fixtures/'V1a.mp4'),
        '-f','lavfi','-i','sine=frequency=440:sample_rate=32000:duration=3',
        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-ac','2',
        '-t','3','-movflags','+faststart',str(path)],check=True)
    assets['A1'] = {'id':'A1','kind':'video','path':str(path),'source':'motion-a',
                    'expected_frames':72,'audio':True,'soundtrack':'440 Hz synthetic tone, stereo 32 kHz'}
    for a in assets.values():
        if a['kind'] != 'video':
            continue
        d = probe(a['path'], True)
        s = next(x for x in d['streams'] if x['codec_type'] == 'video')
        a.update(sha256=sha(a['path']), frames=int(s['nb_read_frames']),
                 fps=s['avg_frame_rate'], width=s['width'], height=s['height'],
                 duration=max(float(d['format']['duration']), *(float(x.get('duration',0)) for x in d['streams'])),
                 media=d, vae_frames=a['expected_frames']-(a['expected_frames']-5)%17)
        a['latent_t'] = (a['vae_frames']-5)//17*5+2
        assert a['frames'] == a['expected_frames'] and a['duration'] <= 3.000001
        subprocess.run(['ffmpeg','-v','error','-y','-i',a['path'],'-frames:v','1',
                        str(fixtures/(a['id']+'.jpg'))],check=True)
        a['thumbnail'] = f'fixtures/{a["id"]}.jpg'
    cases = make_cases()
    for c in cases:
        validate_case(c, assets)
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(0)
    memory = nv.nvmlDeviceGetMemoryInfo(handle)
    environment = {'gpu_name': nv.nvmlDeviceGetName(handle), 'gpu_uuid': nv.nvmlDeviceGetUUID(handle),
        'driver': nv.nvmlSystemGetDriverVersion(), 'vram_total': memory.total,
        'vram_idle': memory.used, 'cgroup': cg_memory(), 'python': sys.version,
        'uname': list(os.uname()), 'nvcc': subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version']).decode(),
        'ffmpeg': subprocess.check_output(['ffmpeg','-version']).decode().splitlines()[0],
        'ffprobe': subprocess.check_output(['ffprobe','-version']).decode().splitlines()[0],
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    assert not nv.nvmlDeviceGetComputeRunningProcesses(handle), 'GPU is not exclusive'
    nv.nvmlShutdown()
    model = Path(args.model).resolve()
    model_files = {}
    for p in sorted(model.rglob('*')):
        if p.is_file():
            st = p.stat()
            model_files[str(p.relative_to(model))] = {'size':st.st_size,'mtime_ns':st.st_mtime_ns,
                                                      'sha256':sha(p)}
    assert any(k.startswith('Ref2VA/') for k in model_files)
    assert any(k.startswith('FL2VA/') for k in model_files)
    save(ROOT/'model-identity.json',model_files)
    save(ROOT/'environment.json',environment)
    settings = {'protocol':PROTOCOL,'model':str(model),'model_identity':digest(model_files),
                'binary_sha256':sha('bin/h3cli'),'gpu_uuid':environment['gpu_uuid'],
                'driver':environment['driver'],'prompt':PROMPT,'weight_mode':'resident',
                'dtype':'original BF16','attention':'default dense','vae':'full',
                'reuse':1,'core_reuse':1,'max_evaluations':6,'max_video_seconds':6,
                'sample_interval_seconds':.05,'gpu_headroom_bytes':4*GIB,
                'host_headroom_bytes':10*GIB,'process_limit_bytes':110000000000}
    manifest = {'settings':settings,'assets':assets,'cases':cases}
    manifest['identity'] = digest(manifest)
    save(ROOT/'manifest.json',manifest)
    print('Prepared and frozen',len(cases),'cases:',manifest['identity'],flush=True)


def command(c, manifest, output):
    args = ['./bin/h3cli','-d',manifest['settings']['model'],'-p',c['prompt'],
            '--width',str(c['width']),'--height',str(c['height']),
            '--frames',str(c['frames']),'--steps',str(c['steps']),'--seed',str(c['seed']),
            '--reuse','1','--core-reuse','1','--cuda-device','0',
            '--cuda-weight-mode','resident','--cuda-attention','default',
            '--cuda-denoise-quant','off','--ref-image-size',c['image_size']]
    for key in c['references']:
        a = manifest['assets'][key]
        flag = '--ref-image' if a['kind']=='image' else '--ref-video' if a['audio'] else '--ref-silent-video'
        args += [flag,a['path']]
    return args + ['-o',str(output)]


class Monitor(threading.Thread):
    def __init__(self, path, settings):
        super().__init__(daemon=True)
        import pynvml
        import psutil
        self.nv,self.ps = pynvml,psutil
        self.nv.nvmlInit()
        self.handle = self.nv.nvmlDeviceGetHandleByIndex(0)
        assert self.nv.nvmlDeviceGetUUID(self.handle) == settings['gpu_uuid']
        assert not self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle), 'Other GPU process is active'
        self.idle = self.nv.nvmlDeviceGetMemoryInfo(self.handle).used
        self.settings,self.path = settings,path
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.pid = None
        self.error = self.abort = None
        self.peaks = dict(device=0,process=0,rss=0,swap=0,cgroup=0)
        self.samples,self.max_gap = 0,0
        self.gaps = 0
        self.process_counter_available = True
        self.owned = set()
        self.namespace = set()
        self.last_tree = 0
        self.last_tree_data = (0,0)

    def process_tree(self, now):
        if now-self.last_tree < .2:
            return self.last_tree_data
        rss=swap=0
        if self.pid:
            try:
                p=self.ps.Process(self.pid)
                procs=[p]+p.children(recursive=True)
            except self.ps.Error:
                procs=[]
            for p in procs:
                try:
                    if p.pid != self.pid and p.pid not in self.owned:
                        self.owned.add(p.pid)
                    rss += p.memory_info().rss
                    for line in Path(f'/proc/{p.pid}/status').read_text().splitlines():
                        if line.startswith('VmSwap:'): swap+=int(line.split()[1])*1024
                        if line.startswith('NSpid:'): self.namespace.update(map(int,line.split()[1:]))
                except (self.ps.Error,OSError):
                    pass
        self.last_tree=now
        self.last_tree_data=(rss,swap)
        return rss,swap

    def run(self):
        previous=None
        try:
            with gzip.open(self.path,'wt') as f:
                while not self.stop_event.is_set():
                    start=time.monotonic()
                    m=self.nv.nvmlDeviceGetMemoryInfo(self.handle)
                    ps=self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle)
                    processes=[]
                    for p in ps:
                        used=getattr(p,'usedGpuMemory',None)
                        if used is None or used >= (1<<63):
                            used=None
                            self.process_counter_available=False
                        processes.append({'pid':p.pid,'used_bytes':used})
                    rss,swap=self.process_tree(start)
                    cg=cg_memory()
                    available=self.ps.virtual_memory().available
                    row={'monotonic':start,'device_bytes':m.used,'device_free':m.free,
                         'processes':processes,'rss':rss,'swap':swap,'cgroup':cg,
                         'host_available':available,'h3_pid':self.pid}
                    f.write(json.dumps(row,separators=(',',':'))+'\n')
                    self.samples+=1
                    for k,v in [('device',m.used),('process',sum(p['used_bytes'] or 0 for p in processes)),
                                ('rss',rss),('swap',swap),('cgroup',cg['used'] or 0)]:
                        self.peaks[k]=max(self.peaks[k],v)
                    if previous is not None:
                        gap=start-previous
                        self.max_gap=max(self.max_gap,gap)
                        self.gaps+=gap>.1
                    previous=start
                    if m.free < self.settings['gpu_headroom_bytes']:
                        self.abort='GPU headroom below 4 GiB'
                    if available < self.settings['host_headroom_bytes']:
                        self.abort='Host headroom below 10 GiB'
                    if cg['available_estimate'] is not None and cg['available_estimate']<self.settings['host_headroom_bytes']:
                        self.abort='Cgroup headroom below 10 GiB'
                    if rss+swap >= self.settings['process_limit_bytes']:
                        self.abort='Process tree footprint reached 110 GB'
                    # PID namespaces can hide the host PID from /proc. Preserve
                    # the mismatch and detect competing compute processes by count.
                    if self.pid and len(processes)>1:
                        self.abort='Unexpected concurrent GPU compute processes'
                    self.ready.set()
                    self.stop_event.wait(max(0.,.05-(time.monotonic()-start)))
        except Exception as e:
            self.error=f'{type(e).__name__}: {e}'
            self.ready.set()

    def finish(self):
        self.stop_event.set()
        self.join(timeout=10)
        self.nv.nvmlShutdown()
        return {'idle_bytes':self.idle,'peaks_bytes':self.peaks,'samples':self.samples,
                'max_gap_seconds':self.max_gap,'gaps_over_100ms':self.gaps,
                'error':self.error,'guard_reason':self.abort,
                'process_counter_available':self.process_counter_available,
                'owned_child_pids':sorted(self.owned),'namespace_pids':sorted(self.namespace)}


def terminate(p):
    try: os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL)
        p.wait(timeout=10)


def validate_output(path, c, logpath):
    if not path.exists():
        return {'valid':False,'errors':['No output file']}
    result={'sha256':sha(path),'errors':[]}
    try:
        d=probe(path,True)
        result['probe']=d
        s=next(x for x in d['streams'] if x['codec_type']=='video')
        result['frames']=int(s['nb_read_frames'])
        duration_limit=362/24+1/24 if c.get('scope')=='long-362' else 10.
        for ok,message in [((s['width'],s['height'])==(c['width'],c['height']),'Wrong resolution'),
                           (result['frames']==c['frames'],'Wrong frame count'),
                           (s['avg_frame_rate']=='24/1','Wrong frame rate'),
                           (float(d['format']['duration'])<=duration_limit,'Container exceeds authorized output duration'),
                           (any(x['codec_type']=='audio' for x in d['streams']),'Missing generated audio')]:
            if not ok: result['errors'].append(message)
        for s in d['streams']:
            if float(s.get('duration',0))>duration_limit: result['errors'].append('Stream exceeds authorized output duration')
        with Path(logpath).open('w') as log:
            p=subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],
                             stdout=log,stderr=log)
        if p.returncode: result['errors'].append('Full media decode failed')
    except (KeyError,ValueError,StopIteration,subprocess.CalledProcessError) as e:
        result['errors'].append(f'Media probe failed: {e}')
    result['valid']=not result['errors']
    return result


def compare(row, rows):
    b=[r for r in rows if r.get('case',{}).get('baseline') and r.get('status')=='complete'
       and r['case']['baseline_key']==row['case']['baseline_key']
       and r.get('manifest_identity')==row.get('manifest_identity')]
    if row.get('status')!='complete' or not b:
        return {'available':False,'reason':'Render incomplete or no completed matching baseline'}
    t=statistics.median(r['wall_seconds'] for r in b)
    g=statistics.median(r['max_vram_gib'] for r in b)
    lo,hi=min(r['wall_seconds'] for r in b),max(r['wall_seconds'] for r in b)
    return {'available':True,'baseline_ids':[r['case']['id'] for r in b],
            'baseline_samples':len(b),'baseline_wall_seconds':t,'baseline_vram_gib':g,
            'baseline_range_seconds':[lo,hi],'baseline_unstable':(hi-lo)/t>.1,
            'baseline_complete':len(b)==(1 if row['case']['phase'] in ('smoke','sample') else 3),
            'wall_delta_seconds':row['wall_seconds']-t,'wall_over_baseline':row['wall_seconds']/t,
            'wall_overhead_percent':100*(row['wall_seconds']/t-1),
            'vram_delta_gib':row['max_vram_gib']-g}


def records(include_superseded=False):
    rows=[json.loads(p.read_text()) for p in sorted(ROOT.glob('R*/attempt-*/metrics.json'))]
    path=ROOT/'retries.json'
    retired={r['prior_directory'] for r in json.loads(path.read_text())} if path.exists() else set()
    return rows if include_superseded else [r for r in rows if r['artifact_directory'] not in retired]


def freeze_check(m):
    assert m==load_manifest(),'Manifest changed'
    assert sha('bin/h3cli')==m['settings']['binary_sha256'],'Binary changed'
    for a in m['assets'].values():
        assert sha(a['path'])==a['sha256'],f'Asset changed: {a["id"]}'
    model=Path(m['settings']['model'])
    identity=json.loads((ROOT/'model-identity.json').read_text())
    assert digest(identity)==m['settings']['model_identity']
    for k,v in identity.items():
        st=(model/k).stat()
        assert (st.st_size,st.st_mtime_ns)==(v['size'],v['mtime_ns']),f'Model changed: {k}'


def run_environment(c):
    env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
    # The renderer's optional diagnostic budget accepts only the literal 6.
    # Final ten-step renders use the normal CLI, bounded by this runner's
    # frozen ten-step contract. Production memory guards are unaffected.
    if c['phase'] in ('smoke','sample'):
        if c['phase']=='sample':assert 1<=c['steps']<=2
        env['H3_TEST_MAX_EVALUATIONS']='6'
    else:assert c['phase']=='main' and c['steps']==10
    return env


def run_case(c, m, timeout):
    import psutil
    prior=list((ROOT/c['id']).glob('attempt-*'))
    assert all((p/'metrics.json').exists() for p in prior),'Unfinished attempt requires explicit recovery'
    attempt=1+max((int(p.name.split('-')[1]) for p in prior),default=0)
    directory=ROOT/c['id']/f'attempt-{attempt}'
    directory.mkdir(parents=True,exist_ok=False)
    out=directory/'output.mp4'
    argv=command(c,m,out)
    assert c['steps']<=m['settings']['max_evaluations']
    env=run_environment(c)
    row={'case':c,'manifest_identity':m['identity'],'protocol':PROTOCOL,
         'case_identity':digest(c),'extension_identity':m.get('extension_identity'),
         'final_identity':m.get('final_identity'),
         'sampling_identity':m.get('sampling_identity'),
         'runner_sha256':sha(__file__),'argv':argv,'cwd':str(Path.cwd()),
         'controlled_environment':{'H3_TEST_MAX_EVALUATIONS':env.get('H3_TEST_MAX_EVALUATIONS')},
         'runner_max_evaluations':m['settings']['max_evaluations'],
         'timeout_seconds':timeout,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'artifact_directory':str(directory.relative_to(ROOT)), 'status':'running'}
    save(directory/'request.json',row)
    with (ROOT/'commands.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    monitor=Monitor(directory/'telemetry.jsonl.gz',m['settings'])
    monitor.start()
    assert monitor.ready.wait(10),'Telemetry initialization timeout'
    if monitor.error or monitor.abort:
        row.update(status='blocked',block_reason=monitor.error or monitor.abort,
                   telemetry=monitor.finish(),wall_seconds=None,max_vram_gib=None)
        save(directory/'metrics.json',row)
        return row
    reason=None
    with (directory/'stdout.log').open('wb') as stdout,(directory/'stderr.log').open('wb') as stderr:
        start=time.monotonic_ns()
        p=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
        monitor.pid=p.pid
        while p.poll() is None:
            if monitor.error or monitor.abort:
                reason=monitor.error or monitor.abort
                terminate(p)
            elif (time.monotonic_ns()-start)/1e9>timeout:
                reason='Timed out'
                terminate(p)
            else:
                time.sleep(.05)
        # h3cli normally waits for media children; account for any surviving owned
        # descendants before ending the full wall interval.
        for pid in list(monitor.owned):
            try:
                child=psutil.Process(pid)
                if child.status()!=psutil.STATUS_ZOMBIE: child.wait(timeout=10)
            except psutil.NoSuchProcess:
                pass
            except psutil.TimeoutExpired:
                reason=reason or 'Media child remained alive after CLI exit'
                try: os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError: pass
        wall=(time.monotonic_ns()-start)/1e9
    released=False
    for _ in range(100):
        memory=monitor.nv.nvmlDeviceGetMemoryInfo(monitor.handle)
        processes=monitor.nv.nvmlDeviceGetComputeRunningProcesses(monitor.handle)
        if not processes and memory.used<=monitor.idle+256*(1<<20):
            released=True
            break
        time.sleep(.1)
    telemetry=monitor.finish()
    log=(directory/'stderr.log').read_text(errors='replace')
    if re.search(r'memory safety:|out of memory|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation',log,re.I):
        reason=reason or 'Application memory guard/allocation failure; review affected branch'
    counts=re.findall(r'\bdenoise\s+(\d+)\s*/\s*(\d+)',log)
    completed=max((int(n) for n,total in counts if int(total)==c['steps']),default=0)
    vid=[{'reference_index':int(i),'pipeline':recipe,'normalized_frames':int(n),
          'vae_frames':int(v),'latent_t':int(t),'posterior_seed':seed}
         for i,recipe,n,v,t,seed in re.findall(r'reference video (\d+): pipeline=(\S+) normalized-frames=(\d+) VAE-frames=(\d+) latent-T=(\d+) posterior-seed=(\S+)',log)]
    validation=validate_output(out,c,directory/'decode.log')
    row.update(wall_seconds=wall,returncode=p.returncode,completed_steps=completed,
               termination_reason=reason,telemetry=telemetry,gpu_released=released,
               max_vram_gib=telemetry['peaks_bytes']['device']/GIB,
               process_peak_vram_gib=telemetry['peaks_bytes']['process']/GIB if telemetry['process_counter_available'] else None,
               video_preprocessing=vid,validation=validation,
               finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               image_geometry_source='host-computed; CLI does not print per-image canvas',
               packed_sequence_observed=None)
    passed=(p.returncode==0 and validation['valid'] and completed==c['steps']
            and not reason and not telemetry['error'] and released)
    row['status']='complete' if passed else 'failed'
    row['error_tail']=log[-4000:] if not passed else None
    if out.exists(): row['output']=str(out.relative_to(ROOT))
    save(directory/'metrics.json',row)
    print(c['id'],row['status'],f'{wall:.2f}s {row["max_vram_gib"]:.3f}GiB',flush=True)
    return row


def run(args):
    import fcntl
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'runner.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=load_manifest()
    assert not m.get('sampling_identity') or args.phase=='smoke', \
        'The unfinished ten-step matrix was superseded; use cuda_multi_reference_sample.py within its frozen deadline'
    freeze_check(m)
    done={r['case']['id']:r for r in records()}
    for r in done.values():
        assert r['manifest_identity']==m['identity'],'Stale result belongs to another manifest'
        assert r['case']==next(c for c in m['cases'] if c['id']==r['case']['id']),'Case definition changed'
    count=0
    for c in m['cases']:
        if c['id'] in done or (args.phase!='all' and c['phase']!=args.phase): continue
        if args.only and c['id'] not in args.only.split(','): continue
        validate_case(c,m['assets'])
        if c['phase']=='main':
            assert m.get('final_identity') and c['steps']==10,'Freeze the ten-step final matrix before main renders'
        # Estimate from matching two-step controls; baseline calibration is
        # conservative, and the chosen value is frozen in each request record.
        smoke=[r for r in done.values() if r['case'].get('baseline') and r['case']['phase']=='smoke'
               and r['case']['width']==c['width'] and r.get('status')=='complete']
        estimate=max((r['wall_seconds'] for r in smoke),default=300.)
        timeout=args.timeout or (max(3000.,estimate*40) if c.get('scope')=='long-362'
                                 else max(1500.,estimate*14) if c['phase']=='main' else 3600.)
        row=run_case(c,m,timeout)
        done[c['id']]=row
        count+=1
        save(ROOT/'results.json',{'manifest_identity':m['identity'],'records':list(done.values())})
        if row.get('termination_reason') or not row.get('gpu_released',False):
            print('STOP: inspect resource/telemetry/cleanup condition before continuing',flush=True)
            break
        if args.limit and count>=args.limit: break


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--model',default=os.environ.get('H3_MODEL_DIR','models/MiniMax-H3'))
    r=sub.add_parser('run')
    r.add_argument('--phase',choices=['all','smoke','main'],default='all')
    r.add_argument('--only')
    r.add_argument('--limit',type=int)
    r.add_argument('--timeout',type=float)
    sub.add_parser('report')
    sub.add_parser('extend-362')
    sub.add_parser('freeze-final')
    args=p.parse_args()
    if args.action=='prepare': prepare(args)
    elif args.action=='run': run(args)
    elif args.action=='extend-362': extend()
    elif args.action=='freeze-final': freeze_final()
    else:
        from cuda_multi_reference_report import report
        report()


if __name__=='__main__':
    main()
