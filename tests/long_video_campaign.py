"""Fixed, serial 256x256 CUDA acceptance cases; at most 464 frames per artifact."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
from long_video_guard import plan, run_case, inspect_state, inspect_video

PROMPT='A red ceramic teapot on a wooden table, soft studio light.'

def digest(path):
    data=Path(path).read_bytes();nv,na=struct.unpack_from('<2Q',data,72)
    assert len(data)==160+nv+na
    return hashlib.sha256(data[160:160+nv]).hexdigest(),hashlib.sha256(data[160+nv:]).hexdigest()

def script_plan(data):
    rows=data['segments']; by_name={}
    for s in rows:
        cmd=s['command']
        def value(name):return cmd[cmd.index('--'+name)+1]
        context=int(value('continue-context')) if '--continue-context' in cmd else (39 if '--continue-from' in cmd else 0)
        plan('script',int(value('width')),int(value('height')),int(value('frames')),context=context,steps=int(value('steps')))
        by_name[s['name']]=s['delivered']
    for merged in data['merged']:
        if sum(by_name[n] for n in merged['segments'])>480:raise ValueError('script merge exceeds 20 seconds')
    return plan('script',256,256,max(s['frames'] for s in rows))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('native','conditions','lifecycle','continuation','script','api','http','recovery','soundtrack'))
    ap.add_argument('--model',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source',type=Path);ap.add_argument('--preview-model',required=True)
    ap.add_argument('--binary',type=Path,default=Path('bin/h3cli'))
    a=ap.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    base=[str(a.binary.resolve()),'-d',a.model,'--offline','--verbose']
    records=[]
    def run(name,p,flags=(),media=True,command=None):
        output=a.out/(name+'.mp4')
        cmd=command or base+list(map(str,flags))+['-o',str(output)]
        records.append(run_case(cmd,p,output,media=media));return output
    def native(name,frames=464,extra=(),capture=True):
        return run(name,plan('native',256,256,frames),['-p',PROMPT,'--width','256','--height','256',
            '--frames',frames,'--steps','2','--seed','42','--save-av-state',a.out/(name+'.h3av'),
            *(['--save-upscale-state',a.out/(name+'.h3up')] if capture else []),*extra])
    def inspect_all():
        for path in a.out.rglob('*.mp4'):
            if path.stat().st_size:inspect_video(path)
        for path in a.out.rglob('*.h3av'):inspect_state(path)
    with open('/tmp/h3-long-video-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.mode=='native':
            native('native')
            subprocess.run(['bin/long_video_state',str(a.out/'native.h3up')],check=True)
            p=inspect_state(a.out/'native.h3av')
            run('decode',p,['--decode-av-state',a.out/'native.h3av'])
            assert hashlib.sha256((a.out/'native.mp4').read_bytes()).digest()==hashlib.sha256((a.out/'decode.mp4').read_bytes()).digest()
            run('preview',p,['--decode-av-state',a.out/'native.h3av','--preview-vae','--preview-vae-model',a.preview_model])
        elif a.mode=='conditions':
            ff=os.environ.get('H3_FFMPEG','ffmpeg')
            for name,color in (('first','red'),('last','blue')):
                subprocess.run([ff,'-v','error','-f','lavfi','-i',f'color=c={color}:s=256x256','-frames:v','1',str(a.out/(name+'.png'))],check=True)
            subprocess.run([ff,'-v','error','-f','lavfi','-i','testsrc2=size=256x256:rate=24',
                '-f','lavfi','-i','sine=frequency=440:sample_rate=44100','-t','2','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(a.out/'reference.mp4')],check=True)
            subprocess.run([ff,'-v','error','-i',str(a.out/'reference.mp4'),'-vn',str(a.out/'reference.wav')],check=True)
            inspect_video(a.out/'reference.mp4',max_frames=360)
            native('anchors',extra=['--first-frame',a.out/'first.png','--last-frame',a.out/'last.png'])
            native('references',extra=['--ref-image',a.out/'first.png','--ref-audio',a.out/'reference.wav','--ref-silent-video',a.out/'reference.mp4'])
        elif a.mode=='soundtrack':
            assert a.source;inspect_video(a.source,max_frames=360)
            native('soundtrack',extra=['--ref-video',a.source.resolve()])
        elif a.mode=='lifecycle':
            assert a.source;p=inspect_state(a.source)
            run('pause',p,['-p',PROMPT,'--width','256','--height','256','--frames','464','--steps','2','--seed','42',
                '--stop-after-step','1','--save-sampler-state',a.out/'paused.h3sample'],False)
            run('resume',p,['--resume-sampler-state',a.out/'paused.h3sample','--save-av-state',a.out/'resumed.h3av'])
            assert digest(a.source)==digest(a.out/'resumed.h3av')
        elif a.mode=='continuation':
            native('source',447)
            for mode in ('hard','bridge'):
                extra=['--continue-from',a.out/'source.h3av','--continue-context','447','--continue-mode',mode]
                run(mode,plan('continuation',256,256,464,context=447),['-p',PROMPT,'--width','256','--height','256',
                    '--frames','464','--steps','2','--seed','42','--save-av-state',a.out/(mode+'.h3av'),
                    '--save-sampler-state',a.out/(mode+'.h3sample'),*extra])
                state=inspect_state(a.out/(mode+'.h3av'));state['delivered']=17
                run(mode+'-decode',state,['--decode-av-state',a.out/(mode+'.h3av')])
                assert (a.out/(mode+'.mp4')).read_bytes()==(a.out/(mode+'-decode.mp4')).read_bytes()
                # Complete hard prefix equals source mixed with the seeded initial noise;
                # bridge has an adaptable front and an exact tail. The checkpoint verifier
                # compares the protected tensors against the saved initial state.
                subprocess.run(['bin/long_video_state',str(a.out/(mode+'.h3sample')),str(a.out/'source.h3av')],check=True)
                listing=a.out/(mode+'.concat');listing.write_text("file 'source.mp4'\nfile '"+mode+".mp4'\n")
                ff=os.environ.get('H3_FFMPEG','ffmpeg')
                subprocess.run([ff,'-v','error','-f','concat','-safe','0','-i',str(listing),'-c','copy',str(a.out/(mode+'-merged.mp4'))],check=True)
                info=inspect_video(a.out/(mode+'-merged.mp4'));v=next(s for s in info['streams'] if s['codec_type']=='video');assert int(v['nb_read_frames'])==464
        elif a.mode=='script':
            text='### long-shot\n#size 256x256\n#frames 464\n#steps 2\n'+PROMPT+'\n'
            for name,body in (('long',text),('seeds','#seed-set 1,2\n### one\n#size 256x256\n#frames 22\n#steps 2\n'+PROMPT+'\n### two\n'+PROMPT+'\n')):
                source=a.out/(name+'.txt');source.write_text(body);directory=a.out/name
                cmd=base+['--render',str(source),'--output-dir',str(directory)]
                data=json.loads(subprocess.check_output(cmd+['--dry-run'],text=True))
                p=script_plan(data);run(name,p,media=False,command=cmd)
        elif a.mode=='api':
            run('api',plan('api',256,256,464),media=False,command=['bin/long_video_api',a.model,str(a.out/'source.h3up')])
        elif a.mode=='http':
            run('http',plan('http',256,256,464),media=False,command=['python3','tests/long_video_http.py','--model',a.model,'--out',str(a.out/'http')])
        elif a.mode=='recovery':
            assert a.source;p=inspect_state(a.source)
            run('recovery',p,command=['bin/long_video_recovery',a.model,str(a.source),str(a.out/'preserved.mp4'),str(a.out/'recovery.mp4')])
            run('preview-cancel',p,media=False,command=['bin/long_video_recovery','preview',a.model,str(a.source),a.preview_model])
            # Deliberately preserved non-media sentinel is not a video artifact.
            (a.out/'preserved.mp4').rename(a.out/'preserved.sentinel')
        inspect_all()
    (a.out/'result.json').write_text(json.dumps(dict(status='passed',mode=a.mode,records=records),indent=2)+'\n')

if __name__=='__main__':main()
