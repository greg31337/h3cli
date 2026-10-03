#!/usr/bin/env python3
"""Replay completed Sage states through both builds using decoder assets only."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse,hashlib,json,re,subprocess
from pathlib import Path
from attention_run import run,sha
from continuation_metrics import state
from attention_media import inspect_state_media

def decoded_hash(path,kind):
    command=['ffmpeg','-v','error','-i',str(path)]
    command+=['-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-'] if kind=='video' else ['-map','0:a:0','-f','f32le','-acodec','pcm_f32le','-']
    child=subprocess.Popen(command,stdout=subprocess.PIPE);digest=hashlib.sha256();count=0
    for block in iter(lambda:child.stdout.read(1<<20),b''):digest.update(block);count+=len(block)
    if child.wait():raise RuntimeError('FFmpeg decode comparison failed')
    return {'sha256':digest.hexdigest(),'bytes':count}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',default='./bin/h3cli');p.add_argument('--non-sage-binary',required=True)
    p.add_argument('--decoder-root',required=True,type=Path);p.add_argument('--preview-model',required=True)
    p.add_argument('--input',nargs='+',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);results=[]
    for variant in ('FL2VA','Ref2VA'):
        transformer=a.decoder_root/variant/'transformer'
        if list(transformer.glob('*.safetensors')):raise RuntimeError('decoder-only fixture contains DiT weights')
    for source in a.input:
        presentation=Path(str(source)+'.presentation').read_text()
        version=int(presentation.splitlines()[0].split()[1])
        assert version == 8,'unsupported presentation provenance'
        width,height=map(int,re.search(r'^output (\d+) (\d+)$',presentation,re.M).groups())
        trim=int(re.search(r'^trim (\d+) ',presentation,re.M)[1])
        frames=state(source)[0][2]-trim
        for vae in ('preview','full'):
            outputs={}
            for build,binary in [('sage',a.binary),('non-sage',a.non_sage_binary)]:
                name=f'{source.stem}-{vae}-{build}';media=a.output/(name+'.mp4');record=a.output/(name+'.json')
                command=[binary,'-d',str(a.decoder_root),'--decode-av-state',str(source),'--profile','-o',str(media)]
                if vae=='preview':command+=['--preview-vae','--preview-vae-model',a.preview_model]
                if record.exists():
                    row=json.loads(record.read_text())
                    if row['argv']!=command or row['binary_sha256']!=sha(binary) or row['returncode']:raise RuntimeError('changed or failed decode record')
                else:row=run(name,command,a.output,1800,{})
                if row['returncode']:raise RuntimeError(f'decode failed; inspect {row["log"]}')
                outputs[build]={'record':row,'media':inspect_state_media(media,source,width,height,frames),
                    'video':decoded_hash(media,'video'),'audio':decoded_hash(media,'audio')}
            exact=all(outputs['sage'][domain]==outputs['non-sage'][domain] for domain in ('video','audio'))
            results.append({'source':str(source),'source_sha256':sha(source),'presentation_version':version,'vae':vae,'exact_across_builds':exact,'outputs':outputs})
            (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            if not exact:raise RuntimeError('decoded video/audio differs across Sage and non-Sage builds')
    (a.output/'complete.json').write_text(json.dumps({'states':len(a.input),'pairs':len(results),'pass':True})+'\n')

if __name__=='__main__':main()
