#!/usr/bin/env python3
"""Fixed 14-video latent-upscale comparison with immutable, resumable accounting."""
import argparse
from array import array
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import threading
import time
from cuda_reference_regression import fingerprint, sha, source_files, write, PROMPT
from cuda_adaptive_subblock import model_metadata, runtime
from cuda_sglang import NVML
from test_sampler_file import entries

IDS = ['L0', 'D0', 'P0', 'I4', 'U0', 'U2', 'U4']
PAIRS = [('768p', [672, 384], [1344, 768]), ('1088p', [960, 544], [1920, 1088])]


def manifest(path):
    m = json.loads(path.read_text())
    assert [m[k] for k in ('schema', 'recipe', 'frames', 'fps', 'source_steps', 'direct_steps', 'seed')] == [1, 1, 90, 24, 50, 50, 42]
    assert m['prompt'] == PROMPT and m['precision'] == 'dense BF16'
    assert [(p['id'], p['source'], p['target']) for p in m['pairs']] == PAIRS
    methods = ['source', 'direct', 'pixel-lanczos', 'latent-bilinear', 'learned', 'learned', 'learned']
    for pair in m['pairs']:
        assert [c['id'] for c in pair['cases']] == IDS
        assert [c['method'] for c in pair['cases']] == methods
        assert [c['evaluations'] for c in pair['cases']] == [50, 50, 0, 4, 0, 2, 4]
        assert [c['start_video_sigma'] for c in pair['cases']] == [None, None, None, .25, None, .25, .25]
        assert [c['output'] for c in pair['cases']] == [f'{pair["id"]}/{i}/video.mp4' for i in IDS]
    assert m['execution']['ordinary_max_evaluations'] == 6
    assert m['execution']['serial'] and m['execution']['no_successful_repeats']
    assert m['execution']['render_timeout_seconds'] == 7200 and m['execution']['transfer_timeout_seconds'] == 1200
    return m


def budget(kind):
    return '50' if kind in ('source', 'direct') else '6'


def av(path):
    b = path.read_bytes()
    assert b[:8] == b'H3AV\r\n\x1a\n' and struct.unpack_from('<I', b, 8)[0] == 3
    w, h, frames, vt, lh, lw, at, vc, ac, channels = struct.unpack_from('<10I', b, 24)
    assert (frames, vt, at, vc, ac, channels) == (90, 27, 150, 24, 32, 2)
    assert (w, h) == (lw*16, lh*16)
    nv, na = 24*vt*lh*lw, 64*at
    assert struct.unpack_from('<2Q', b, 72) == (nv*4, na*4) and len(b) == 160+(nv+na)*4
    assert hashlib.sha256(b[:128]+b[160:]).digest() == b[128:160]
    values = array('f'); values.frombytes(b[160:]); assert all(map(math.isfinite, values))
    return dict(width=w, height=h, audio_sha256=hashlib.sha256(b[160+nv*4:]).hexdigest(),
                video_sha256=hashlib.sha256(b[160:160+nv*4]).hexdigest())


def media(path, size, env):
    probe = json.loads(subprocess.check_output([env.get('H3_FFPROBE', 'ffprobe'), '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(path)], env=env))
    v, = [s for s in probe['streams'] if s['codec_type'] == 'video']
    a, = [s for s in probe['streams'] if s['codec_type'] == 'audio']
    assert (v['width'], v['height'], int(v['nb_read_frames']), v['avg_frame_rate']) == (*size, 90, '24/1')
    assert a['channels'] == 2 and int(a['sample_rate']) == 32000
    assert abs(float(v['duration'])-3.75) < 1/24 and abs(float(a['duration'])-3.75) <= 1025/32000
    decoded = subprocess.run([env.get('H3_FFMPEG', 'ffmpeg'), '-v', 'error', '-i', str(path), '-f', 'null', '-'], env=env, capture_output=True, text=True)
    assert decoded.returncode == 0 and not decoded.stderr.strip(), decoded.stderr
    return probe


def transitions(log, count, refinement=False):
    rows = [json.loads(line.split('h3_experiment ', 1)[1]) for line in log.splitlines() if 'h3_experiment {' in line]
    assert [r['step'] for r in rows] == list(range(count)), 'missing or repeated transition'
    assert all(r['total'] == count and r['evaluated'] == 1 and r['blocks'] == 50 and r['sparse_calls'] == r['quant_calls'] == 0 for r in rows)
    if refinement:
        assert all(r['audio_sigma'] == 0 for r in rows) and rows[0]['video_sigma'] == .25
    return rows


def phases(log):
    result = {}
    for line in log.splitlines():
        if 'h3_upscale_phase {' in line:
            r = json.loads(line.split('h3_upscale_phase ', 1)[1])
            result[r['phase']] = result.get(r['phase'], 0)+r['seconds']
    return result


def completed(stage):
    results = []
    for d in sorted(stage.glob('attempt-*')):
        p = d/'result.json'
        assert p.exists(), 'unrecorded attempt needs inspection: '+str(d)
        r = json.loads(p.read_text())
        if r['passed']:
            for name, digest in r['artifacts'].items():
                assert sha(d/name) == digest, 'changed successful artifact: '+str(d/name)
            results.append((d, r))
        else:
            assert r.get('returncode') != 0 and not (d/'video.mp4').exists(), 'completed output needs inspection; never rerender it'
    assert len(results) <= 1, 'duplicate successful job'
    return results[0] if results else None


def accounting(records, pair):
    def wall(name): return records[f'{pair}/{name}']['wall_seconds']
    low = records[f'{pair}/L0']
    excluded = sum(v for k, v in low['phase_seconds'].items() if k.startswith(('audio VAE', 'video VAE', 'FFmpeg')))
    source = max(0., low['wall_seconds']-excluded)
    out = {}
    for case in IDS:
        extra = 0.
        if case in ('U0', 'U2', 'U4'): extra += wall('shared/learned')
        if case == 'I4': extra += wall('shared/bilinear')
        if case in ('I4', 'U2', 'U4'): extra += wall(f'shared/{case}-init')
        if case in ('U2','U4'): extra += records[f'{pair}/shared/I4-init'].get('shared_noise_creation_seconds',0.)
        later = wall(case)+extra
        out[case] = dict(observed_render_seconds=wall(case), shared_and_initialization_seconds=extra,
                         later_job_seconds=later, source_without_delivery_seconds=source,
                         end_to_end_seconds=later+(low['wall_seconds'] if case=='P0' else source if case not in ('L0', 'D0') else 0.))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'model', 'weights', 'manifest', 'gate', 'out'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--resume', action='store_true');p.add_argument('--dry-run', action='store_true')
    a = p.parse_args();m = manifest(a.manifest)
    if a.dry_run:
        print(json.dumps([dict(pair=pair['id'], **c, budget=budget(c['method'])) for pair in m['pairs'] for c in pair['cases']], indent=2));return
    root, model, out = a.source.resolve(), a.model.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=a.resume)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = os.environ.copy()
        assert not any(k.startswith('H3_TEST_') and k != 'H3_TEST_MAX_EVALUATIONS' for k in env)
        env = {k: v for k, v in env.items() if not k.startswith('H3_PROFILE')}
        env |= {'H3_TEST_MAX_EVALUATIONS': '6', 'H3_EXPERIMENT_TRACE': '1', 'H3_EXPERIMENT_TIMING': '1', 'H3_VERBOSE': '1'}
        gate = json.loads(a.gate.read_text());source_hash = fingerprint(source_files(root))
        assert gate['passed'] and len(gate['files']) == 204 and gate['source_sha256'] == source_hash
        assert sha(a.weights) == '4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6'
        binary = root/'bin/upscale_campaign'
        identity = dict(schema=1, source_sha256=source_hash, binary_sha256=sha(binary), manifest_sha256=sha(a.manifest), weights_sha256=sha(a.weights),
                        gate_sha256=sha(a.gate), model_metadata=model_metadata(model), runtime=runtime(env))
        frozen = out/'identity.json'
        if frozen.exists(): assert json.loads(frozen.read_text()) == identity, 'changed frozen build/environment'
        else: write(frozen, identity);(out/'manifest.json').write_bytes(a.manifest.read_bytes())
        records = {};ledger = dict(complete=False, human_visual_review='pending', planned=14, cases=[], stages=records)

        def immutable():
            assert fingerprint(source_files(root)) == source_hash and sha(binary) == identity['binary_sha256']
            assert sha(a.manifest) == identity['manifest_sha256'] and model_metadata(model) == identity['model_metadata']

        def job(name, kind, args, products, evaluations=None, size=None):
            immutable();stage = out/name;stage.mkdir(parents=True, exist_ok=True)
            success = completed(stage)
            if success:
                d, record = success
            else:
                assert not any((stage/f).exists() for f in products), 'unrecorded published output needs inspection'
                d = stage/f'attempt-{len(list(stage.glob("attempt-*")))+1:03d}';d.mkdir()
                command = [str(x).replace('{out}', str(d)) for x in args]
                record = dict(passed=False, command=command, kind=kind, budget=budget(kind), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                write(d/'result.json', record)
                assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU is busy'
                monitor = NVML();samples=[];errors=[];stop=threading.Event();begin=time.monotonic()
                with (d/'run.log').open('x') as log, (d/'memory.jsonl').open('x') as memory:
                    child = subprocess.Popen(command, cwd=root, env=env|{'H3_TEST_MAX_EVALUATIONS': budget(kind)}, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    def sample():
                        try:
                            while not stop.is_set():
                                r = monitor.sample(child.pid)|dict(elapsed_seconds=time.monotonic()-begin);samples.append(r)
                                memory.write(json.dumps(r)+'\n');memory.flush();stop.wait(1)
                        except BaseException as e: errors.append(str(e))
                    worker=threading.Thread(target=sample);worker.start()
                    try: record['returncode'] = child.wait(timeout=7200 if size else 1200)
                    except BaseException as e:
                        record['error']=str(e);os.killpg(child.pid,signal.SIGTERM)
                        try: child.wait(timeout=10)
                        except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
                    finally:
                        stop.set();worker.join();record.update(wall_seconds=time.monotonic()-begin, telemetry_errors=errors)
                        write(d/'result.json', record)
                        with (out/'attempts.jsonl').open('a') as f: f.write(json.dumps(dict(stage=name, attempt=d.name, **record))+'\n')
                assert record.get('returncode') == 0 and samples and not errors, name+' failed'
                log=(d/'run.log').read_text();record['phase_seconds']=phases(log)
                record['shared_noise_creation_seconds']=sum(json.loads(line.split(' ',1)[1])['seconds'] for line in log.splitlines()
                    if line.startswith('h3_upscale_noise_persist ') or (line.startswith('h3_upscale_noise_time ') and not json.loads(line.split(' ',1)[1])['reused']))
                if evaluations is not None:
                    steps=transitions(log,evaluations,kind=='refinement');write(d/'steps.json',steps)
                    record['transitions']=len(steps);record['blocks']=sum(s['blocks'] for s in steps)
                if size: write(d/'ffprobe.json',media(d/'video.mp4',size,env))
                if 'final.h3av' in products: record['av']=av(d/'final.h3av')
                record.update(peak_vram_bytes=max(s['gpu_used_bytes'] for s in samples),
                              peak_host_rss_bytes=max(s.get('process_rss_bytes',0) for s in samples),
                              artifacts={f:sha(d/f) for f in products}, passed=True)
                write(d/'result.json',record)
                with (out/'attempts.jsonl').open('a') as f: f.write(json.dumps(dict(stage=name, attempt=d.name, validated=True))+'\n')
            for f, digest in record['artifacts'].items():
                dest=stage/f
                if dest.exists(): assert sha(dest)==digest, 'stale publication'
                else: os.link(d/f,dest)
            record=record|dict(attempt=str(d.relative_to(out)));records[name]=record
            write(stage/'result.json',record);write(out/'ledger.json',ledger);immutable()
            print(json.dumps(dict(stage=name, wall_seconds=record['wall_seconds'], passed=True)),flush=True)
            return stage

        av_products=['final.h3av','final.h3av.presentation']
        for pair in m['pairs']:
            name=pair['id'];base=out/name;source_path=base/'L0/source.h3up'
            for case, kind, size in [('L0','source',pair['source']),('D0','direct',pair['target'])]:
                job(f'{name}/{case}',kind,[binary,'generate',model,'{out}',*size,kind,m['prompt']],
                    ['video.mp4',*av_products]+(['source.h3up'] if case=='L0' else []),50,size)
            job(f'{name}/P0','pixel',[env.get('H3_FFMPEG','ffmpeg'),'-v','error','-i',base/'L0/video.mp4','-vf',
                f'scale={pair["target"][0]}:{pair["target"][1]}:flags=lanczos','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','copy','{out}/video.mp4'],['video.mp4'],0,pair['target'])
            for method, recipe in [('learned',1),('bilinear',2)]:
                job(f'{name}/shared/{method}','transfer',[binary,'transfer',model,source_path,a.weights,'{out}',recipe],av_products,0)
            shared_noise=base/'shared/I4-init/initial.h3sample'
            for case, method, k in [('I4','bilinear',4),('U0','learned',0),('U2','learned',2),('U4','learned',4)]:
                transferred=base/f'shared/{method}/final.h3av';input_path=transferred;input_kind='av'
                if k:
                    job(f'{name}/shared/{case}-init','initialize',[binary,'initialize',model,source_path,transferred,
                        '-' if case=='I4' else shared_noise,'{out}',k],['initial.h3sample']+(['shared-noise.f32'] if case=='I4' else []),0)
                    input_path=base/f'shared/{case}-init/initial.h3sample';input_kind='sampler'
                    noise=next(p[-1] for p in entries(input_path.read_bytes()) if p[0]==21)
                    assert hashlib.sha256(noise).hexdigest()==sha(base/'shared/I4-init/shared-noise.f32')
                job(f'{name}/{case}','refinement' if k else 'decode',[binary,'render',model,input_path,'{out}',input_kind],['video.mp4',*av_products],k,pair['target'])
                assert records[f'{name}/{case}']['av']['audio_sha256']==records[f'{name}/L0']['av']['audio_sha256']
            for case in ('L0','D0','I4','U0','U2','U4'):
                job(f'{name}/verify-{case}-pcm','verification',[binary,'pcm',model,base/f'{case}/final.h3av','{out}/audio.f32'],['audio.f32'],0)
                if case!='D0': assert sha(base/f'verify-{case}-pcm/audio.f32')==sha(base/'verify-L0-pcm/audio.f32')
            for case in IDS: ledger['cases'].append(dict(pair=name,id=case,output=f'{name}/{case}/video.mp4',sha256=sha(base/f'{case}/video.mp4')))
            ledger.setdefault('costs',{})[name]=accounting(records,name);write(out/'ledger.json',ledger)
        assert len(ledger['cases'])==14;ledger['complete']=True;write(out/'ledger.json',ledger)


if __name__=='__main__': main()
