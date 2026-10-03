#!/usr/bin/env python3
from source_tree import source_path
"""Fresh before/after FL2VA and short/15-second Ref2VA parity on the installed model.

Uses all 50 DiT layers and reduced step counts to test arithmetic invariance.
The small FL2VA case compares fresh end-to-end output. Normal-resolution cases
also capture freshly encoded text, then replay ONLY text before DiT to isolate
pre-existing cold Qwen variability. Pixels and VAE conditions are never replayed.
These renders are regression fixtures, not output-quality benchmarks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from source_tree import build_path
import re
import shutil
import subprocess
import tarfile
import time
from continuation_regression import HELPER

from source_tree import copy_source_tree

ROOT=Path(__file__).resolve().parents[1]


def instrument(directory):
    path=source_path(directory, 'engine.c'); source=path.read_text()
    needle='static char h3_global_error[512];'
    assert source.count(needle)==1
    source=source.replace(needle,needle+HELPER)
    needle='    h3_layout_spec spec ='
    assert source.count(needle)==1
    source=source.replace(needle,'''    regression_buffer("condition-video",condition_video_rows,condition_video_elements*4,0);
    regression_buffer("condition-audio",condition_audio_rows,condition_audio_elements*4,0);
    regression_buffer("fresh-text",text.values,text.tokens*text.width*2,0);
    regression_buffer("text",text.values,text.tokens*text.width*2,1);
    regression_buffer("text",text.values,text.tokens*text.width*2,0);
    if (text.diagnostics) {
        regression_buffer("ids",text.diagnostics->ids,text.diagnostics->tokens*4,0);
        regression_buffer("positions",text.diagnostics->positions,text.diagnostics->tokens*3*4,0);
        regression_buffer("spans",text.diagnostics->spans,text.diagnostics->span_count*2*sizeof(size_t),0);
    } else regression_buffer("ids",ids,token_count*4,0);
'''+needle)
    pattern=r'    (?:if \()?h3_progress_emit\(&progress, "audio VAE", 0, 7\);?(?:\) goto cleanup;)?'
    source,n=re.subn(pattern,lambda m: '''    regression_buffer("video",video,video_count*4,0);
    regression_buffer("audio",audio,audio_count*4,0);
'''+m[0],source)
    assert n==1
    needle='    size_t rgb_count = (size_t)frames.frames * (size_t)frames.height *'
    assert source.count(needle)==1
    source=source.replace(needle,'''    regression_buffer("decoded-rgb",frames.rgb,(size_t)frames.frames*frames.height*frames.width*3*4,0);
    regression_buffer("decoded-pcm",waveform.pcm,(size_t)waveform.samples*waveform.channels*4,0);
'''+needle)
    needle='        size_t condition_offset = 0;'
    assert source.count(needle)==1
    source=source.replace(needle,'''        for (size_t i=0;i<visual_count;i++) {
            char name[64]; snprintf(name,sizeof(name),"pixels-%zu",i);
            regression_buffer(name,condition_pixels[i],(size_t)condition_frames[i]*condition_widths[i]*condition_heights[i]*3*4,0);
        }
'''+needle)
    path.write_text(source)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''): h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',default='b162b9f')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--only',default='fl2va-small,fl2va,ref-short,ref-long')
    p.add_argument('--rebuild',action='store_true')
    args=p.parse_args()
    out=ROOT/'outputs/memory-validation/generation'; out.mkdir(parents=True,exist_ok=True)
    builds={}
    for label in ('before','after'):
        directory=out/label; directory.mkdir(exist_ok=True)
        if args.rebuild or not (args.resume and (build_path(directory, 'h3cli')).exists()):
            if label=='before':
                archive=out/'before.tar'
                subprocess.run(['git','archive','-o',str(archive),args.baseline],cwd=ROOT,check=True)
                with tarfile.open(archive) as tar: tar.extractall(directory,filter='data')
            else:
                copy_source_tree(ROOT, directory)
            instrument(directory)
            with (directory/'build.log').open('w') as log:
                subprocess.run(['make','-j8','all'],cwd=directory,stdout=log,stderr=log,check=True)
        builds[label]=directory
    if args.prepare_only: return
    for name,frames in [('short',48),('long',360)]:
        reference=out/(name+'.mp4')
        if not reference.exists():
            subprocess.run(['ffmpeg','-v','error','-y','-loop','1','-framerate','24',
                '-i',str(ROOT/'inputs/body1.jpg'),'-vf','scale=256:256',
                '-frames:v',str(frames),'-c:v','libx264','-pix_fmt','yuv420p',str(reference)],check=True)
    cases={
        'fl2va-small': (22,4,['--first-frame',str(ROOT/'inputs/face1.jpg')]),
        'fl2va': (56,4,['--first-frame',str(ROOT/'inputs/face1.jpg')]),
        'ref-short': (56,4,['--ref-image',str(ROOT/'inputs/2.jpg'),'--ref-silent-video',str(out/'short.mp4')]),
        'ref-long': (243,2,['--ref-silent-video',str(out/'long.mp4')]),
    }
    record_path=out/'results.json'
    records=json.loads(record_path.read_text()) if args.resume and record_path.exists() else {}
    for case in args.only.split(','):
        frames,steps,refs=cases[case]
        for label in ('before','after'):
            name=case+'-'+label
            if args.resume and name in records: continue
            directory=builds[label]; base=out/name
            command=[str(build_path(directory, 'h3cli')),'-d',str(ROOT/'models/MiniMax-H3'),'-p',
                'A woman smiles.' if case=='fl2va-small' else 'A woman looks at the camera and slowly turns her head.',
                '--width','32' if case=='fl2va-small' else '256','--height','32' if case=='fl2va-small' else '256','--frames',str(frames),'--steps',str(steps),
                '--seed','72',*refs,'-o',str(base)+'.mp4']
            env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
            env['H3_REGRESSION_DUMP']=str(base)
            if label=='after' and case!='fl2va-small':
                env['H3_REGRESSION_REPLAY']=str(out/(case+'-before'))
            print('running',name,flush=True); start=time.monotonic()
            with Path(str(base)+'.log').open('w') as log:
                subprocess.run(command,cwd=directory,env=env,stdout=log,stderr=log,check=True)
            hashes={f.name[len(name)+1:]:digest(f) for f in out.glob(name+'.*') if f.suffix!='.log'}
            records[name]={'command':command,'seconds':time.monotonic()-start,'hashes':hashes,
                'binary_sha256':digest(build_path(directory, 'h3cli')), 'text_replayed':label=='after' and case!='fl2va-small'}
            record_path.write_text(json.dumps(records,indent=2)+'\n')
            print('completed',name,round(records[name]['seconds'],1),'seconds',flush=True)
        before={k:v for k,v in records[case+'-before']['hashes'].items() if k!='fresh-text'}
        after={k:v for k,v in records[case+'-after']['hashes'].items() if k!='fresh-text'}
        assert before==after,case+' parity failed'
        print('ok: exact pixels, conditioning, text, latents, decoded RGB/PCM and MP4:',case,flush=True)


if __name__=='__main__': main()
