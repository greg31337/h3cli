#!/usr/bin/env python3
"""Current-feature cleanup qualification; no historical numerical oracle.

Runs bounded real jobs sequentially, records complete wall times and media
metadata, and compares fresh state round trips. Requires an existing CUDA build.
"""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def adapter(model, output):
    """Small nonzero rank-one functional adapter; not a trained quality model."""
    root = model / 'FL2VA/transformer'
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    name = 'blocks.0.attn.qkv_proj.weight'
    with (root / index[name]).open('rb') as f:
        header = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
    rows, cols = header[name]['shape']
    a = struct.pack('<f', .002) * cols
    b = struct.pack('<f', .02) * rows
    tensors = {
        'blocks.0.attn.qkv_proj.lora_A.weight': dict(dtype='F32', shape=[1, cols], data_offsets=[0, len(a)]),
        'blocks.0.attn.qkv_proj.lora_B.weight': dict(dtype='F32', shape=[rows, 1], data_offsets=[len(a), len(a)+len(b)]),
        '__metadata__': {'description': 'Synthetic rank-one cleanup functional fixture'},
    }
    data = json.dumps(tensors, separators=(',', ':')).encode()
    data += b' ' * (-len(data) % 8)
    output.write_bytes(struct.pack('<Q', len(data)) + data + a + b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--preview-model', type=Path, required=True)
    p.add_argument('--image-vae', type=Path, required=True)
    p.add_argument('--upscale-model', type=Path, required=True)
    p.add_argument('--lora-cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='Reuse only successful jobs from the identical binary')
    p.add_argument('--rerun', action='append', default=[], help='Repeat a named job when resuming; retain its previous attempt record')
    a = p.parse_args()
    source, out, model = a.source.resolve(), a.out.resolve(), a.model.resolve()
    out.mkdir(parents=True, exist_ok=a.resume)
    binary = source / 'bin/h3cli'
    identity = sha(binary)
    rows = []
    comparisons = []
    complete = False
    env = dict(os.environ, H3_TEST_MAX_EVALUATIONS='6', H3_EXPERIMENT_TRACE='1')
    # Launcher bookkeeping must be identical for saves and later replay. The
    # conservative state identity records otherwise-unrecognized H3 variables.
    for key in ('H3_REFERENCE_MODEL','H3_REFERENCE_REGRESSION_OUT'):
        env.pop(key,None)
    fixtures = source / 'tests/fixtures/cuda-reference'
    first, image = fixtures / 'first.png', source / 'inputs/face1.jpg'
    last = source / 'inputs/2.jpg'
    video, wav = fixtures / 'reference.mp4', out / 'reference-2p5.wav'
    if not wav.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-stream_loop', '-1', '-i', str(video),
                        '-t', '2.5', '-vn', '-ar', '32000', '-ac', '2', str(wav)], check=True)
    lora = out / 'rank-one.safetensors'
    if not lora.exists():
        adapter(model, lora)
    inputs = [first, last, image, video, wav, lora]
    (out/'inputs.json').write_text(json.dumps({str(f): sha(f) for f in inputs}, indent=2)+'\n')
    if a.resume and (out/'result.json').exists():
        previous = json.loads((out/'result.json').read_text())
        if previous['binary_sha256'] != identity:
            raise RuntimeError('resume requires the identical binary')
        rows = previous['jobs']
    completed = {r['name']: r for r in rows if r['passed'] and r['name'] not in a.rerun}

    def publish():
        result = dict(binary_sha256=identity, complete=complete, jobs=rows, comparisons=comparisons,
                      passed=complete and bool(rows) and all(r['passed'] for r in rows) and all(c['passed'] for c in comparisons))
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        cards = ['<!doctype html><meta charset="utf-8"><title>Cleanup render review</title>',
                 '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto}video,img{max-width:100%;max-height:600px}pre{white-space:pre-wrap}article{margin:2em 0;border-top:1px solid #aaa}</style>',
                 '<h1>Cleanup functional render review</h1><p>Each job uses at most six denoiser evaluations. Synthetic LoRA is a functional fixture. Visual acceptance is not inferred.</p>']
        for r in rows:
            cards += [f'<article><h2>{html.escape(r["name"])}</h2><p>{r["wall_seconds"]:.2f} s total wall time · {"passed" if r["passed"] else "failed"}</p>']
            if r.get('media') and Path(r['media']).exists():
                name = html.escape(Path(r['media']).name)
                cards.append(f'<img src="{name}">' if name.endswith('.png') else f'<video controls preload="metadata" src="{name}"></video>')
            cards += [f'<pre>{html.escape(" ".join(r["argv"]))}</pre><a href="{html.escape(r["name"])}.log">log</a></article>']
        (out/'review.html').write_text('\n'.join(cards))

    def run(name, flags, shape=(640,480,124), steps=6, operation=False, deliver=True, still=False):
        media = out/(name+('.png' if still else '.mp4'))
        cmd = [str(binary), '-d', str(model)]
        if not operation:
            prompt = 'A person explores a sunny garden, moving naturally as the camera slowly glides sideways. Warm daylight and soft outdoor ambience.'
            cmd += ['-p', prompt, '--width', str(shape[0]), '--height', str(shape[1]), '--steps', str(steps), '--seed', '42']
            if not still:
                cmd += ['--frames', str(shape[2])]
        cmd += list(map(str, flags)) + ['--profile']
        if '--inspect-upscale-state' not in flags:
            cmd += ['-o', str(media)]
        if name in completed and completed[name]['argv'] == cmd:
            return media
        previous = next((r for r in rows if r['name']==name), None)
        if previous:
            history = out/'attempts'/name
            history.mkdir(parents=True,exist_ok=True)
            attempt=history/('%03d'%(len(list(history.iterdir()))+1))
            attempt.mkdir()
            (attempt/'result.json').write_text(json.dumps(previous,indent=2)+'\n')
            for suffix in ('.log','.memory.csv'):
                path=out/(name+suffix)
                if path.exists():shutil.copy2(path,attempt/path.name)
        start = time.monotonic()
        record = dict(name=name, argv=cmd, passed=False, media=str(media) if deliver else None)
        monitor_log = (out/(name+'.memory.csv')).open('w')
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-lms', '200'], stdout=monitor_log, stderr=subprocess.DEVNULL)
        try:
            with (out/(name+'.log')).open('w') as log:
                proc = subprocess.run(cmd, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            record['wall_seconds'] = time.monotonic()-start
            record['returncode'] = proc.returncode
            if proc.returncode:
                raise RuntimeError('render exited '+str(proc.returncode))
            if deliver:
                streams = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(media)]))['streams']
                v = next(s for s in streams if s['codec_type']=='video')
                if (int(v['width']),int(v['height'])) != shape[:2]:
                    raise RuntimeError('unexpected delivered dimensions')
                if not still:
                    if int(v['nb_read_frames']) != shape[2] or v['r_frame_rate'] != '24/1':
                        raise RuntimeError('unexpected delivered frame count/rate')
                    audio = next((s for s in streams if s['codec_type']=='audio'), None)
                    if audio is None:
                        raise RuntimeError('missing generated audio')
                    if abs(float(audio['duration'])-shape[2]/24) > .12:
                        raise RuntimeError('unexpected audio duration')
                subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(media),'-f','null','-'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                record.update(streams=streams, sha256=sha(media))
            log_text = (out/(name+'.log')).read_text()
            trace = [json.loads(line.split('h3_experiment ',1)[1]) for line in log_text.splitlines() if line.startswith('h3_experiment ')]
            record['steps'] = trace
            record['adaptive_decisions'] = [line for line in log_text.splitlines() if 'h3cli: adaptive step=' in line]
            if 'subblock' in flags and not (trace and sum(t['sparse_calls'] for t in trace)):
                raise RuntimeError('SubBlock had no sparse kernel activity')
            if '--adaptive-cache' in flags and len(record['adaptive_decisions']) != steps:
                raise RuntimeError('missing adaptive policy decisions')
            if '--reuse' in flags and not (trace and sum(t['evaluated'] for t in trace)<steps):
                raise RuntimeError('reuse did not skip any transformer evaluations')
            record['passed'] = True
        except (OSError, RuntimeError, subprocess.SubprocessError, KeyError, StopIteration) as exc:
            record['error'] = str(exc)
        finally:
            monitor.terminate(); monitor.wait(timeout=10); monitor_log.close()
        values = [int(v) for v in (out/(name+'.memory.csv')).read_text().splitlines() if v.strip().isdigit()]
        record['sampled_peak_gpu_mib'] = max(values,default=0)
        record.setdefault('wall_seconds',time.monotonic()-start)
        record['validation_seconds'] = time.monotonic()-start-record['wall_seconds']
        rows[:] = [r for r in rows if r['name'] != name]
        rows.append(record)
        publish()
        print(name, record['passed'], round(record['wall_seconds'],2), record.get('error',''), flush=True)
        return media

    def av(name): return out/(name+'.h3av')
    def state(name): return out/(name+'.h3sample')
    def compare(name, left, right, latent=False):
        try:
            if latent:
                # AV schema 3 has a fixed 160-byte header. Metadata is checked by native readers.
                assert left.read_bytes()[160:] == right.read_bytes()[160:]
            else:
                def content(path):
                    return subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0','-f','framemd5','-'])
                assert content(left) == content(right)
            comparisons.append(dict(name=name, passed=True))
        except (OSError, AssertionError, subprocess.SubprocessError) as exc:
            comparisons.append(dict(name=name, passed=False, error=str(exc)))
        publish()

    cond = out/'text.h3cond'
    run('R01-text', ['--save-av-state',av('R01-text'),'--save-sampler-state',state('R01-text'),'--save-conditioning',cond])
    run('R01-upscale-source', ['--save-upscale-state',out/'R01-text.h3up',
        '--save-av-state',av('R01-upscale-source')])
    compare('baseline upscale source export',av('R01-text'),av('R01-upscale-source'),latent=True)
    run('R02-first', ['--first-frame',first])
    run('R03-last', ['--last-frame',last])
    run('R04-anchors', ['--first-frame',first,'--last-frame',last])
    ref = ['--ref-image',image,'--ref-image-size','max']
    refcond = out/'reference.h3cond'
    run('R05-image', ref+['--save-av-state',av('R05-image'),'--save-conditioning',refcond])
    run('R06-images', ['--ref-image',image,'--ref-image',last,'--ref-image-size','high'])
    run('R07-audio', ['--ref-image',image,'--ref-audio',wav])
    run('R08-video', ['--ref-video',video])
    run('R09-silent', ['--ref-silent-video',video])
    run('R09-soundtrack', ['--ref-video-audio',video,wav])
    run('R10-mixed', ['--ref-image',image,'--ref-video',video,'--ref-audio',wav])
    full = run('R11-full', ['--decode-av-state',av('R01-text')], operation=True)
    run('R11-preview', ['--decode-av-state',av('R01-text'),'--preview-vae','--preview-vae-model',a.preview_model], operation=True)
    compare('AV full decode content',out/'R01-text.mp4',full)
    for name, flags, cache, control in [('text',[],cond,'R01-text'),('reference',ref,refcond,'R05-image')]:
        run('R12-'+name+'-pause',flags+['--stop-after-step','3','--save-sampler-state',state(name)],deliver=False)
        run('R12-'+name+'-resume',['--resume-sampler-state',state(name),'--save-av-state',av(name)],operation=True)
        compare(name+' pause/resume',av(control),av(name),latent=True)
        run('R12-'+name+'-cache',flags+['--load-conditioning',cache,'--save-av-state',av(name+'-cache')])
        compare(name+' conditioning reuse',av(control),av(name+'-cache'),latent=True)
    for mode in ('hard','bridge'):
        # Request 56 internal frames; the 39-frame prefix is trimmed on delivery.
        run('R13-'+mode,['--continue-from',av('R01-text'),'--continue-mode',mode,'--frames','56'],shape=(640,480,17))
    for name, flags in [('text',[]),('anchors',['--first-frame',first,'--last-frame',last])]:
        up=out/(name+'.h3up')
        run('R14-'+name+'-source',flags+['--save-upscale-state',up,'--save-av-state',av('source-'+name)],shape=(320,256,22))
        run('R14-'+name+'-inspect',['--inspect-upscale-state',up],operation=True,deliver=False)
        opts=['--upscale-state',up,'--upscale-model',a.upscale_model,'--upscale-refine-steps','2']
        run('R14-'+name+'-upscale',opts+['--save-av-state',av('up-'+name)],operation=True,shape=(640,512,22))
        run('R14-'+name+'-pause',opts+['--stop-after-step','1','--save-sampler-state',state('refine-'+name)],operation=True,deliver=False)
        run('R14-'+name+'-resume',['--resume-sampler-state',state('refine-'+name),'--save-av-state',av('refine-'+name)],operation=True,shape=(640,512,22))
        compare(name+' refinement resume',av('up-'+name),av('refine-'+name),latent=True)
        try:
            def audio_latent(path):
                data=path.read_bytes();return data[160+struct.unpack_from('<Q',data,72)[0]:]
            equal=audio_latent(av('source-'+name))==audio_latent(av('up-'+name))
            comparisons.append(dict(name=name+' upscale audio preservation',passed=equal))
        except OSError as exc:
            comparisons.append(dict(name=name+' upscale audio preservation',passed=False,error=str(exc)))
    # Clean-source export and resumable checkpoint capture are separate public
    # operations. Exercise completed-sampler import with its own fresh control.
    run('R14-text-sampler-source',['--save-sampler-state',state('up-text'),
        '--save-av-state',av('source-sampler')],shape=(320,256,22))
    compare('upscale source vs completed sampler control',av('source-text'),av('source-sampler'),latent=True)
    run('R14-sampler-import',['--upscale-state',state('up-text'),'--upscale-import-sampler','--upscale-model',a.upscale_model,'--upscale-refine-steps','0'],operation=True,shape=(640,512,22))
    still=out/'still.safetensors'
    original=run('R15-still',['--still','--image-vae',a.image_vae,'--save-still-latent',still],still=True,shape=(512,512,1))
    decoded=run('R15-still-decode',['--decode-still-latent',still,'--image-vae',a.image_vae],operation=True,still=True,shape=(512,512,1))
    compare('still decode',original,decoded)
    run('R15-lora',['--lora',lora,'--lora-cache',a.lora_cache])
    for reuse in (2,3): run('R16-reuse-'+str(reuse),['--reuse',str(reuse),'--save-sampler-state',state('reuse-'+str(reuse))])
    sparse=['--cuda-attention','subblock','--subblock-sparsity','.75','--subblock-warmup','2']
    run('R16-adaptive-subblock',sparse+['--adaptive-cache','conservative','--adaptive-cache-warmup','2','--save-sampler-state',state('adaptive')])
    run('R16-subblock-reference',ref+sparse+['--save-sampler-state',state('subblock-ref')])
    for quant,attention in [('fp8','sage2++'),('nvfp4','sage3')]:
        flags=['--cuda-denoise-quant',quant,'--cuda-attention',attention]
        run('R17-'+quant,flags+['--save-av-state',av(quant)])
        run('R17-'+quant+'-decode',['--decode-av-state',av(quant)],operation=True)
        run('R17-'+quant+'-pause',flags+['--stop-after-step','3','--save-sampler-state',state(quant)],deliver=False)
        run('R17-'+quant+'-resume',['--resume-sampler-state',state(quant),'--save-av-state',av(quant+'-resume')],operation=True)
        compare(quant+' pause/resume',av(quant),av(quant+'-resume'),latent=True)
    run('R18-large',ref+sparse,shape=(1344,768,124))
    for name, checkpoint in [('reuse-2','reuse-2'),('reuse-3','reuse-3'),
                             ('adaptive-subblock','adaptive'),('subblock-reference','subblock-ref')]:
        decoded=run('R16-'+name+'-resume',['--resume-sampler-state',state(checkpoint)],operation=True)
        compare(name+' completed checkpoint delivery',out/('R16-'+name+'.mp4'),decoded)
    complete = True
    publish()
    return 0 if all(r['passed'] for r in rows) and all(c['passed'] for c in comparisons) else 1


if __name__ == '__main__':
    raise SystemExit(main())
