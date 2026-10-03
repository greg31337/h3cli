#!/usr/bin/env python3
from source_tree import source_path
"""Isolate existing multimodal-encoder variability from continuation regressions.

Build pinned original and current source in isolated output directories. Capture
original text/reference conditioning and replay it through the current engine.
Assert bit-identical final video/audio latents and encoded MP4 for every mode.
Instrumentation exists ONLY in the isolated test copies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from source_tree import build_path
import shutil
import subprocess
import tarfile
import time

from source_tree import copy_source_tree

ROOT=Path(__file__).resolve().parents[1]
HELPER=r'''
static void regression_buffer(const char *name, void *data, size_t bytes, int replay) {
    const char *prefix=getenv(replay ? "H3_REGRESSION_REPLAY" : "H3_REGRESSION_DUMP");
    if (!prefix || !data || !bytes) return;
    char path[4096]; snprintf(path,sizeof(path),"%s.%s",prefix,name);
    FILE *f=fopen(path,replay?"rb":"wb");
    if (!f) { fprintf(stderr,"regression cannot open %s\n",path); abort(); }
    size_t n=replay?fread(data,1,bytes,f):fwrite(data,1,bytes,f);
    if (n!=bytes || (replay && fgetc(f)!=EOF)) { fprintf(stderr,"regression shape mismatch %s\n",path); abort(); }
    fclose(f);
}
'''

def instrument(directory):
    path=source_path(directory, 'engine.c'); source=path.read_text()
    source=source.replace('static char h3_global_error[512];','static char h3_global_error[512];'+HELPER)
    before='''    regression_buffer("condition-video",condition_video_rows,condition_video_elements*4,1);
    regression_buffer("condition-audio",condition_audio_rows,condition_audio_elements*4,1);
    regression_buffer("text",text.values,text.tokens*text.width*2,1);
    regression_buffer("condition-video",condition_video_rows,condition_video_elements*4,0);
    regression_buffer("condition-audio",condition_audio_rows,condition_audio_elements*4,0);
    regression_buffer("text",text.values,text.tokens*text.width*2,0);
'''
    assert source.count('    h3_layout_spec spec =')==1
    source=source.replace('    h3_layout_spec spec =',before+'    h3_layout_spec spec =')
    needle='    h3_progress_emit(&progress, "audio VAE", 0, 7);'
    assert source.count(needle)==1
    source=source.replace(needle,'''    regression_buffer("video",video,video_count*4,0);
    regression_buffer("audio",audio,audio_count*4,0);
'''+needle)
    path.write_text(source)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',default='948c748')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/continuation-validation/frozen-regression')
    parser.add_argument('--only',default='t2va,fl2va,image-ref2va,video-ref2va,audio-ref2va')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    from run_continuation import prepare_fixtures
    prepare_fixtures()
    original=out/'original'; current=out/'current'
    if not original.exists():
        original.mkdir(); archive=out/'baseline.tar'
        subprocess.run(['git','archive','-o',str(archive),args.baseline],cwd=ROOT,check=True)
        with tarfile.open(archive) as tar: tar.extractall(original,filter='data')
        instrument(original)
    current.mkdir(exist_ok=True)
    copy_source_tree(ROOT, current)
    instrument(current)
    for directory in [original,current]:
        with (directory/'build.log').open('w') as log:
            subprocess.run(['make','-j8','all'],cwd=directory,stdout=log,stderr=log,check=True)
    cases=json.loads((ROOT/'docs/continuation-baseline.json').read_text())['cases']
    record_path=out/'results.json'
    records=json.loads(record_path.read_text()) if args.resume and record_path.exists() else {}
    for name in args.only.split(','):
        if args.resume and records.get(name,{}).get('passed'): continue
        command=cases[name].copy()
        path_flags=['-d','--model-dir','--first-frame','--last-frame','--ref-image','--ref-video','--ref-audio']
        for i,arg in enumerate(command[:-1]):
            if arg in path_flags: command[i+1]=str((ROOT/command[i+1]).resolve())
        record={}
        for label,directory in [('original',original),('current',current)]:
            base=out/(name+'-'+label)
            cmd=command.copy();cmd[0]=str(build_path(directory, 'h3cli'));cmd[cmd.index('-o')+1]=str(base)+'.mp4'
            env=os.environ.copy();env.pop('H3_PROFILE',None)
            env['H3_REGRESSION_DUMP']=str(base)
            if label=='current': env['H3_REGRESSION_REPLAY']=str(out/(name+'-original'))
            else: env.pop('H3_REGRESSION_REPLAY',None)
            print('running frozen regression',name,label,flush=True);start=time.monotonic()
            with Path(str(base)+'.log').open('w') as log:
                subprocess.run(cmd,cwd=directory,stdout=log,stderr=log,env=env,check=True)
            hashes={}
            for extension in ['condition-video','condition-audio','text','video','audio','mp4']:
                path=Path(str(base)+'.'+extension)
                if path.exists(): hashes[extension]=hashlib.sha256(path.read_bytes()).hexdigest()
            record[label]={'hashes':hashes,'seconds':time.monotonic()-start,
                           'command':cmd,'binary_sha256':hashlib.sha256((build_path(directory, 'h3cli')).read_bytes()).hexdigest()}
        record['passed']=record['original']['hashes']==record['current']['hashes']
        records[name]=record;record_path.write_text(json.dumps(records,indent=2))
        assert record['passed'],f'frozen conditioning regression failed: {name}'
        print('ok: exact conditioning, final AV latents and MP4 match:',name,flush=True)

if __name__=='__main__': main()
