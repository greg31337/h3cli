#!/usr/bin/env python3
"""Diagnostic first/middle/last contact sheet; full, native, reference columns."""
import json
from pathlib import Path
import subprocess

root=Path('outputs/preview-vae/metal/quality');records=json.loads((root/'manifest.json').read_text())
rows=[]
for r in records:
    cells=[]
    for mode in ['full','tiny','reference']:
        frames=r['frames'];select=f"select='eq(n,0)+eq(n,{frames//2})+eq(n,{frames-1})',scale=144:192:force_original_aspect_ratio=decrease,pad=144:192:(ow-iw)/2:(oh-ih)/2"
        data=subprocess.check_output(['ffmpeg','-v','error','-i',str(root/f'{r["name"]}-{mode}.mp4'),'-vf',select,'-fps_mode','vfr','-f','rawvideo','-pix_fmt','rgb24','-'])
        size=144*192*3;assert len(data)==size*3
        cells.extend(data[i*size:(i+1)*size] for i in range(3))
    rows.append(b''.join(cell[y*144*3:(y+1)*144*3] for y in range(192) for cell in cells))
subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'1296x{192*len(rows)}','-i','pipe:0','-frames:v','1',str(root/'contact.png')],input=b''.join(rows),check=True)
print('Rows:',', '.join(r['name'] for r in records));print('Columns: original first/middle/last; native first/middle/last; reference first/middle/last')
