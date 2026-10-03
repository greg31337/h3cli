#!/usr/bin/env python3
"""Decode every retained clip and validate authoritative state/presentation pairs."""
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

root=Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/quant-5090')
if '--joins' in sys.argv:
    joins=[]
    for mode in ('fp8','nvfp4'):
        directory=root/f'continuation-{mode}'
        for policy in ('hard','bridge'):
            output=directory/f'joined-{policy}.mp4'
            command=['ffmpeg','-v','error','-i',str(directory/'source.mp4'),'-i',str(directory/f'{policy}.mp4'),
                     '-filter_complex','[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]',
                     '-map','[v]','-map','[a]','-frames:v','141','-t','5.875','-c:v','libx264','-crf','18',
                     '-c:a','aac','-ar','32000','-ac','2','-y',str(output)]
            subprocess.run(command,check=True,timeout=10)
            joins.append(dict(mode=mode,policy=policy,path=str(output),command=command,
                              quality='functional two-step smoke; separate seam review pending'))
    (root/'continuation-joins.json').write_text(json.dumps(joins,indent=2)+'\n')
    (root/'continuation-review.html').write_text('<!doctype html><meta charset="utf-8"><title>Continuation smoke review</title>'
        '<h1>Continuation smoke clips</h1><p>Two-step 64×64 functional clips. The six-case quality approval does not cover these seams. '
        'Each join contains 90 source frames plus 51 new frames; the boundary is at 3.75 seconds.</p>'+''.join(
            f'<h2>{r["mode"]} / {r["policy"]}</h2><video controls src="{Path(r["path"]).relative_to(root)}"></video>' for r in joins))

# CLI records provide expected geometry and bind decode replay to its immutable
# input state, including clips whose output deliberately has no new AV state.
expected={}
for record in root.glob('*.json'):
    if record.name.startswith('.'):continue
    entry=json.loads(record.read_text())
    if not isinstance(entry,dict) or entry.get('status')!='passed':continue
    command=entry.get('command',[])
    if '-o' not in command:continue
    output=Path(command[command.index('-o')+1])
    if '--decode-av-state' in command:
        source=Path(command[command.index('--decode-av-state')+1]);data=source.read_bytes()
        assert hashlib.sha256(data).hexdigest()==entry['file_arguments_sha256'][str(source)],source
        fields={x.split()[0]:x.split()[1:] for x in Path(str(source)+'.presentation').read_text().splitlines()}
        expected[output.resolve()]=[*map(int,fields['output']),struct.unpack_from('<I',data,32)[0]-int(fields['trim'][0])]
    elif all(flag in command for flag in ('--width','--height','--frames')):
        shape=[int(command[command.index(flag)+1]) for flag in ('--width','--height','--frames')]
        if '--continue-from' in command and '--keep-continuation-prefix' not in command:
            shape[2]-=int(command[command.index('--continue-context')+1]) if '--continue-context' in command else 39
        expected[output.resolve()]=shape
records=[]
for path in sorted(root.rglob('*.mp4')):
    streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))['streams']
    video=next(x for x in streams if x['codec_type']=='video');audios=[x for x in streams if x['codec_type']=='audio']
    assert int(video['nb_frames'])>0 and video['r_frame_rate']=='24/1',path
    if 'paused' in path.stem:assert not audios,path
    else:
        assert len(audios)==1,path
        a=audios[0];assert a['sample_rate']=='32000' and a['channels']==2,path
        assert abs(float(video['duration'])-float(a['duration']))<.1,path
    result=subprocess.run(['ffmpeg','-v','info','-i',str(path),'-vf','blackdetect=d=0.04:pix_th=0.005','-f','null','-'],capture_output=True,text=True,timeout=30)
    assert not result.returncode,(path,result.stderr)
    row=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),width=video['width'],height=video['height'],
             frames=int(video['nb_frames']),audio=bool(audios),full_decode=True,
             black_intervals=[x for x in result.stderr.splitlines() if 'black_start' in x])
    if path.resolve() in expected:assert [row['width'],row['height'],row['frames']]==expected[path.resolve()],path
    if path.name.startswith('joined-'):assert [row['width'],row['height'],row['frames']]==[64,64,141],path
    if path.stem=='paused':assert [row['width'],row['height'],row['frames']]==[128,128,22],path
    state=path.with_suffix('.h3av')
    if state.exists():
        data=state.read_bytes();assert data[:8]==b'H3AV\r\n\x1a\n'
        nv,na=struct.unpack_from('<QQ',data,72);assert len(data)==160+nv+na and nv%4==na%4==0
        assert hashlib.sha256(data[:128]+data[160:]).digest()==data[128:160]
        assert all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',data[160:])),state
        sha=hashlib.sha256(data).hexdigest();row['state_sha256']=sha;row['finite_latents']=True
        metadata=Path(str(state)+'.presentation').read_text();fields={x.split()[0]:x.split()[1:] for x in metadata.splitlines()}
        assert fields['state']==[sha]
        assert list(map(int,fields['output']))==[video['width'],video['height']]
        assert int(video['nb_frames'])==struct.unpack_from('<I',data,32)[0]-int(fields['trim'][0])
        row['presentation']=fields
    if path.parent.name=='quality':
        end=int(video['nb_frames'])-1;middle=end//2;poster=path.with_suffix('.jpg')
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-vf',f'select=eq(n\\,0)+eq(n\\,{middle})+eq(n\\,{end}),scale=192:-1,tile=3x1','-frames:v','1','-y',str(poster)],check=True,timeout=30)
        row['poster']=str(poster)
    records.append(row)
(root/'media-index.json').write_text(json.dumps(records,indent=2)+'\n')
print(f'PASS {len(records)} complete media decodes and {sum(x.get("finite_latents",False) for x in records)} finite, checksummed AV states')
