"""Fixed short CUDA acceptance cases. Every subprocess is preflighted and serial."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import shutil
from large_canvas_guard import plan, run_case


def state_shape(path, kind):
    """Inspect dimensions before loading models; production readers validate checksums."""
    with Path(path).open('rb') as f:
        header = f.read(160)
        if header[:8] == b'H3AV\r\n\x1a\n':
            w, h, frames = struct.unpack_from('<3I', header, 24)
        else:
            assert header[:8] in (b'H3SAMPLE', b'H3UPSRC\1'), 'not a current state'
            assert 1 <= struct.unpack_from('<I', header, 20)[0] <= 256
            f.seek(96)
            entries = f.read(struct.unpack_from('<I', header, 20)[0]*72)
            identity = None
            for at in range(0, len(entries), 72):
                section, version, flags, dtype, count, offset, size = struct.unpack_from('<4I3Q', entries, at)
                if section == 1:
                    assert size < 4096
                    f.seek(offset)
                    identity = f.read(size)
            assert identity is not None
            w, h, frames = struct.unpack_from('<3i', identity, 160)
    return plan(kind, w, h, frames)


def script_plan(data):
    total = 0
    cases = []
    for s in data['segments']:
        command = s['command']
        def value(flag):
            return command[command.index('--'+flag)+1]
        continuation = '--continue-from' in command
        p = plan('continuation' if continuation else 'native',
                 int(value('width')), int(value('height')), int(value('frames')))
        total += s['delivered']
        cases.append(p)
    if total > 72:
        raise ValueError('merged script exceeds three seconds')
    return cases, total


def av_digest(path):
    data = Path(path).read_bytes()
    nv, na = struct.unpack_from('<2Q', data, 72)
    assert len(data) == 160+nv+na
    return hashlib.sha256(data[160:160+nv]).hexdigest(), hashlib.sha256(data[160+nv:]).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('portrait', 'crosschunk', 'replay', 'native', 'references', 'continuation', 'script', 'lifecycle', 'api', 'http', 'recovery', 'sanitizer', 'operators', 'conditions', 'mixed', 'package', 'lora-resume'))
    ap.add_argument('--model', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--binary', type=Path, default=Path('bin/h3cli'))
    ap.add_argument('--preview-model', type=Path)
    ap.add_argument('--refine', type=int, choices=(2,3,4), default=4)
    ap.add_argument('--oracle-python')
    ap.add_argument('--upstream', type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    base = [str(a.binary.resolve()), '-d', a.model, '--offline', '--verbose']
    records = []

    def run(name, p, args, media=True, command=None):
        output = a.out/(name+'.mp4')
        cmd = command or base+list(map(str, args))+['-o', str(output)]
        records.append(run_case(cmd, p, output, media=media))
        return output

    def native(name, w, h, frames=22, args=(), steps=2, reference_rows=0):
        p = plan('native',w,h,frames)
        p = plan('native',w,h,frames,rows=p['sequence_rows']+reference_rows)
        return run(name, p,
            ['-p', 'A red ceramic teapot on a wooden table, soft studio light.',
             '--width', w, '--height', h, '--frames', frames, '--steps', steps, '--seed', 42,
             '--save-av-state', a.out/(name+'.h3av'), '--save-upscale-state', a.out/(name+'.h3up'), *args])

    with open('/tmp/h3-large-canvas-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if a.mode in ('portrait','crosschunk'):
                w,h,frames=(1088,1920,22) if a.mode=='portrait' else (1920,1088,39)
                native('native',w,h,frames)
                for k in ((0,2,3,4) if a.mode=='portrait' else (0,2)):
                    run('K'+str(k), plan('upscale',w*2,h*2,frames,refine=k),
                        ['--upscale-state', a.out/'native.h3up', '--upscale-model', a.weights,
                         '--upscale-refine-steps', k, '--save-av-state', a.out/('K'+str(k)+'.h3av')])
                    assert av_digest(a.out/'native.h3av')[1] == av_digest(a.out/('K'+str(k)+'.h3av'))[1]
            elif a.mode == 'replay':
                assert a.source
                p = state_shape(a.source, 'upscale')
                run('decode', p, ['--decode-av-state', a.source])
            elif a.mode == 'native':
                native('small', 256, 128)
                native('repeat', 256, 128)
                assert av_digest(a.out/'small.h3av') == av_digest(a.out/'repeat.h3av')
                native('quality', 1920, 1088, steps=6)
                if a.preview_model:
                    # Preview delivery cannot be a clean upscale source.
                    run('preview', plan('native', 1920, 1088, 22),
                        ['-p', 'A red teapot.', '--width', 1920, '--height', 1088,
                         '--frames', 22, '--steps', 2, '--preview-vae', '--preview-vae-model', a.preview_model])
                p = state_shape(a.out/'quality.h3av', 'native')
                run('decode', p, ['--decode-av-state', a.out/'quality.h3av'])
                if a.preview_model:
                    run('preview-cancel',p,[],False,[str(Path('bin/large_canvas_recovery').resolve()),
                        'preview',a.model,str(a.out/'quality.h3av'),str(a.preview_model)])
                p = plan('native',256,128,22)
                checkpoint = a.out/'paused.h3sample'
                run('pause',p,['-p','A red ceramic teapot on a wooden table, soft studio light.',
                    '--width',256,'--height',128,'--frames',22,'--steps',2,'--seed',42,
                    '--stop-after-step',1,'--save-sampler-state',checkpoint],False)
                state_shape(checkpoint,'native')
                run('resume',p,['--resume-sampler-state',checkpoint,'--save-av-state',a.out/'resumed.h3av'])
                assert av_digest(a.out/'small.h3av')==av_digest(a.out/'resumed.h3av')
            elif a.mode == 'references':
                first=a.out/'large.png'
                portrait=a.out/'portrait.jpg'
                subprocess.run([os.environ.get('H3_FFMPEG','ffmpeg'),'-v','error','-i',
                    'tests/fixtures/cuda-reference/first.png','-vf','scale=3840:2176','-frames:v','1',str(first)],check=True)
                shutil.copyfile('tests/fixtures/cuda-reference/portrait.jpg',portrait)
                for mode in ('match','high','max'):
                    native(mode,1920,1088,args=['--ref-image',first,'--ref-image',portrait,
                        '--ref-image-size',mode],reference_rows=32768)
                native('anchors',1920,1088,args=['--first-frame',first,'--last-frame',portrait],reference_rows=4080)
                first.unlink();portrait.unlink()
                for mode in ('match','high','max','anchors'):
                    old=a.out/(mode+'.h3up');moved=a.out/('relocated-'+mode+'.h3up');old.rename(moved)
                    info=subprocess.check_output(base[:1]+['--inspect-upscale-state',str(moved)],text=True)
                    import re
                    rows=int(re.search(r' rows=(\d+)',info)[1])
                    run('up-'+mode,plan('upscale',3840,2176,22,rows=rows,refine=2),
                        ['--upscale-state',moved,'--upscale-model',a.weights,'--upscale-refine-steps',2,
                         '--save-av-state',a.out/('up-'+mode+'.h3av')])
                    assert av_digest(a.out/(mode+'.h3av'))[1]==av_digest(a.out/('up-'+mode+'.h3av'))[1]
            elif a.mode == 'continuation':
                native('source', 256, 128, 39)
                state_shape(a.out/'source.h3av', 'continuation')
                for mode in ('hard', 'bridge'):
                    p = plan('continuation', 256, 128, 56)
                    args = ['-p', 'The camera gently moves.', '--width', 256, '--height', 128,
                            '--frames', 56, '--steps', 2, '--seed', 42,
                            '--continue-from', a.out/'source.h3av', '--continue-context', 39,
                            '--keep-continuation-prefix', '--save-av-state', a.out/(mode+'.h3av')]
                    if mode == 'bridge':
                        args += ['--continue-mode', 'bridge', '--continue-bridge-steps', 1]
                    run(mode, p, args)
                    state_shape(a.out/(mode+'.h3av'), 'continuation')
                    run(mode+'-decode', p, ['--decode-av-state', a.out/(mode+'.h3av')])
            elif a.mode == 'mixed':
                clip=a.out/'reference.mp4';sound=a.out/'audio.wav';still=a.out/'portrait.jpg'
                plan('native',128,64,39)
                subprocess.run([os.environ.get('H3_FFMPEG','ffmpeg'),'-v','error','-loop','1','-framerate','24',
                    '-i','tests/fixtures/cuda-reference/first.png','-f','lavfi','-i',
                    'sine=frequency=440:sample_rate=32000:duration=1.625','-vf','scale=128:64',
                    '-frames:v','39','-t','1.625','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(clip)],check=True)
                subprocess.run([os.environ.get('H3_FFMPEG','ffmpeg'),'-v','error','-i',str(clip),
                    '-vn','-ar','32000','-ac','2','-af','apad','-t','2',str(sound)],check=True)
                shutil.copyfile('tests/fixtures/cuda-reference/portrait.jpg',still)
                native('mixed',1920,1088,args=['--ref-video-audio',clip,sound,'--ref-audio',sound,
                    '--ref-image',still,'--ref-image-size','high'],reference_rows=32768)
                clip.unlink();sound.unlink();still.unlink()
                source=a.out/'relocated-mixed.h3up';(a.out/'mixed.h3up').rename(source)
                import re
                info=subprocess.check_output(base[:1]+['--inspect-upscale-state',str(source)],text=True)
                rows=int(re.search(r' rows=(\d+)',info)[1])
                run('up-mixed',plan('upscale',3840,2176,22,rows=rows,refine=2),
                    ['--upscale-state',source,'--upscale-model',a.weights,'--upscale-refine-steps',2,
                     '--save-av-state',a.out/'up-mixed.h3av'])
                assert av_digest(a.out/'mixed.h3av')[1]==av_digest(a.out/'up-mixed.h3av')[1]
            elif a.mode == 'script':
                script = a.out/'scenes.txt'
                ref=Path('tests/fixtures/cuda-reference/first.png').resolve()
                script.write_text(f'### first\n#size 1920x1088\n#frames 22\n#steps 2\n#seed 42\n#ref-image "{ref}"\nA red teapot.\n### second\n#ref-image-size high\n#ref-image "{ref}"\nA blue teapot.\n')
                args = base+['--render', str(script), '--output-dir', str(a.out/'render'), '--ref-image-size', 'max']
                data = json.loads(subprocess.check_output(args+['--dry-run'], text=True))
                cases, total = script_plan(data)
                assert total == 44
                p=plan('native',1920,1088,22,rows=cases[0]['sequence_rows']+32768)
                run('script', p, [], media=False, command=args)
                merged = list((a.out/'render').rglob('scenes.mp4'))
                assert len(merged) == 1
                probe = json.loads(subprocess.check_output([os.environ.get('H3_FFPROBE', 'ffprobe'), '-v', 'error',
                    '-count_frames', '-show_streams', '-of', 'json', str(merged[0])], text=True))
                video, = [s for s in probe['streams'] if s['codec_type'] == 'video']
                assert int(video['nb_read_frames']) == 44 and float(video['duration']) < 3
                logs='\n'.join(p.read_text(errors='replace') for p in a.out.rglob('*.log'))
                assert 'reference image 1 size=max' in logs and 'reference image 1 size=high' in logs
            elif a.mode == 'api':
                run('reuse',plan('upscale',3840,2176,22),[],False,
                    [str(Path('bin/large_canvas_api').resolve()),a.model,a.weights,
                     str(a.out/'source.h3up'),str(a.out/'target.h3av')])
            elif a.mode == 'lora-resume':
                assert a.source
                p=state_shape(a.source,'native')
                from test_lora_runtime import tensor_file
                adapter=a.out/'unqualified.safetensors'
                tensor_file(adapter,[('blocks.0.attn.qkv_proj.lora_A.weight',[1,2],[1.,2.],'F32'),
                    ('blocks.0.attn.qkv_proj.lora_B.weight',[6,1],[.5]*6,'F32')])
                run('reject',p,[],False,[str(Path('bin/large_canvas_api').resolve()),
                    'reject-lora-resume',a.model,str(a.source),str(adapter)])
            elif a.mode == 'http':
                run('http',plan('native',1920,1088,22),[],False,
                    ['python3','tests/large_canvas_http.py','--model',a.model,'--out',str(a.out/'http')])
            elif a.mode == 'recovery':
                assert a.source
                p=state_shape(a.source,'upscale')
                run('recovery',p,[],False,[str(Path('bin/large_canvas_recovery').resolve()),a.model,
                    str(a.source),str(a.out/'original.mp4'),str(a.out/'recovered.mp4')])
            elif a.mode == 'sanitizer':
                run('sanitizer',plan('upscale',128,64,22,refine=0),[],False,
                    ['compute-sanitizer','--tool','memcheck','--error-exitcode','99',
                     str(Path('bin/large_canvas_recovery').resolve()),'operators',a.weights,'unused','unused'])
            elif a.mode == 'operators':
                assert a.source and a.oracle_python and a.upstream
                state_shape(a.source,'native')
                p=plan('upscale',256,192,22,refine=0)
                run('oracle',p,[],False,[a.oracle_python,'tests/upscale_reference.py','--short',
                    '--weights',a.weights,'--upstream',str(a.upstream),'--av',str(a.source),
                    '--out',str(a.out/'reference')])
                run('native',p,[],False,[str(Path('bin/upscale_network').resolve()),a.weights,
                    str(a.out/'reference'),str(a.out/'native'),'--short'])
                run('comparison',p,[],False,['python3','tests/upscale_compare.py',
                    '--reference',str(a.out/'reference'),'--native',str(a.out/'native')])
            elif a.mode == 'conditions':
                assert a.source and a.oracle_python and a.upstream
                source_plan=state_shape(a.source,'native')
                p=plan('upscale',2*source_plan['width'],2*source_plan['height'],source_plan['frames'],refine=0)
                run('export',p,[],False,[str(Path('bin/large_canvas_conditions').resolve()),
                    a.weights,str(a.source),str(a.out)])
                run('oracle',p,[],False,[a.oracle_python,'tests/large_canvas_conditions.py',
                    '--weights',a.weights,'--upstream',str(a.upstream),'--directory',str(a.out)])
            elif a.mode == 'package':
                native('native',1920,1088)
                native_checkpoint=a.out/'native-paused.h3sample'
                native_plan=plan('native',1920,1088,22)
                run('native-pause',native_plan,['-p','A red ceramic teapot on a wooden table, soft studio light.',
                    '--width',1920,'--height',1088,'--frames',22,'--steps',2,'--seed',42,
                    '--stop-after-step',1,'--save-sampler-state',native_checkpoint],False)
                state_shape(native_checkpoint,'native')
                run('native-resume',native_plan,['--resume-sampler-state',native_checkpoint,
                    '--save-av-state',a.out/'native-resumed.h3av'])
                assert av_digest(a.out/'native.h3av')==av_digest(a.out/'native-resumed.h3av')
                source=a.out/'native.h3up';checkpoint=a.out/'paused.h3sample'
                state_shape(source,'native')
                p=plan('upscale',3840,2176,22,refine=2)
                run('pause',p,['--upscale-state',source,'--upscale-model',a.weights,
                    '--upscale-refine-steps',2,'--stop-after-step',1,'--save-sampler-state',checkpoint],False)
                state_shape(checkpoint,'upscale')
                run('resume',p,['--resume-sampler-state',checkpoint,'--save-av-state',a.out/'resumed.h3av'])
                run('full',p,['--upscale-state',source,'--upscale-model',a.weights,
                    '--upscale-refine-steps',2,'--save-av-state',a.out/'full.h3av'])
                assert av_digest(a.out/'resumed.h3av')==av_digest(a.out/'full.h3av')
                assert av_digest(a.out/'native.h3av')[1]==av_digest(a.out/'full.h3av')[1]
                state_shape(a.out/'full.h3av','upscale')
                run('decode',p,['--decode-av-state',a.out/'full.h3av'])
            else:
                assert a.source
                info = subprocess.check_output(base[:1]+['--inspect-upscale-state', str(a.source)], text=True)
                import re
                match = re.search(r'target=(\d+)x(\d+) frames=(\d+).*? rows=(\d+)', info)
                assert match
                w, h, frames, rows = map(int, match.groups())
                p = plan('upscale', w, h, frames, rows=rows, refine=a.refine)
                binary = str(Path('bin/upscale_lifecycle').resolve())
                zero = a.out/'zero.h3sample'
                run('prepare', p, [], False, [binary, 'prepare', a.model, str(a.source), a.weights, str(zero), str(a.out/'K0.h3av'),str(a.refine)])
                trace = a.out/'trajectory.f32'
                def lifecycle(name, source, stop, cancel, compare, first):
                    state_shape(source, 'upscale')
                    run(name, p, [], False, [binary, 'run', a.model, str(source), str(a.out/name), str(trace),
                                           str(stop), str(cancel), str(compare), str(first)])
                lifecycle('full', zero, -1, -1, 0, 0)
                previous = zero
                for boundary in range(1, a.refine+1):
                    lifecycle('boundary-'+str(boundary), previous, boundary, -1, 1, boundary-1)
                    previous = a.out/('boundary-'+str(boundary)+'.h3sample')
                assert av_digest(a.out/'full.h3av') == av_digest(a.out/('boundary-'+str(a.refine)+'.h3av'))
                lifecycle('cancel', zero, -1, 1, 1, 0)
                lifecycle('recover', a.out/'cancel.h3sample', -1, -1, 1, 1)
                assert av_digest(a.out/'full.h3av') == av_digest(a.out/'recover.h3av')
                run('decode', p, ['--decode-av-state', a.out/'full.h3av'])
        finally:
            (a.out/'campaign.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
