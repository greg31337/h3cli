#!/usr/bin/env python3
"""Build synchronized review gallery and contact sheets for the three modes."""
import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/scalingfix-validation/generation'
def main():
    summaries={};sections=[]
    for case in ('fl2va','image','video'):
        modes=[m for m in ('legacy','scaled-q','reference') if (OUT/f'{case}-{m}.mp4').exists()]
        if not modes:continue
        sheet=Image.new('RGB',(7*160,len(modes)*184),'white');draw=ImageDraw.Draw(sheet)
        for row,mode in enumerate(modes):
            movie=OUT/f'{case}-{mode}.mp4'
            raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(movie),'-vf','scale=160:160','-f','rawvideo','-pix_fmt','rgb24','-'])
            frames=np.frombuffer(raw,np.uint8).reshape(-1,160,160,3);selected=np.linspace(0,len(frames)-1,7).round().astype(int)
            for col,idx in enumerate(selected):
                sheet.paste(Image.fromarray(frames[idx]),(col*160,row*184+24));draw.text((col*160+4,row*184+5),f'{mode} {idx/24:.2f}s',fill='black')
            dense=Image.new('RGB',(7*160,3*184),'white');dd=ImageDraw.Draw(dense)
            for j,idx in enumerate(np.linspace(0,len(frames)-1,21).round().astype(int)):
                x=(j%7)*160;y=(j//7)*184;dense.paste(Image.fromarray(frames[idx]),(x,y+24));dd.text((x+4,y+5),f'{mode} {idx/24:.2f}s',fill='black')
            dense.save(OUT/f'{case}-{mode}-dense.jpg')
            pcm=np.fromfile(OUT/f'{case}-{mode}.decoded-pcm','<f4').reshape(-1,2)
            change=np.abs(np.diff(frames.astype(np.float32)/255,axis=0)).mean(axis=(1,2,3))
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(movie)]))
            video=next(x for x in probe['streams'] if x['codec_type']=='video');audio=next(x for x in probe['streams'] if x['codec_type']=='audio')
            assert int(video['nb_frames'])==124 and int(audio['channels'])==2 and int(audio['sample_rate'])==32000
            av_delta=abs(float(video['duration'])-float(audio['duration']));assert av_delta<1/24
            summaries[case+'-'+mode]={'frames':len(frames),'mux_av_duration_delta':av_delta,'audio_samples':len(pcm),'audio_seconds':len(pcm)/32000,'audio_peak':float(abs(pcm).max()),'audio_rms':float(np.sqrt(np.mean(pcm*pcm))),
                'frame_difference_mean':float(change.mean()),'frame_difference_max':float(change.max()),'frame_difference_max_time':float(change.argmax()/24)}
        sheet.save(OUT/f'{case}-contact.jpg')
        sections.append('<h2>'+case+'</h2><div class="row">'+''.join(f'<figure><figcaption>{m}</figcaption><video controls muted preload="metadata" src="{case}-{m}.mp4"></video></figure>' for m in modes)+'</div>')
    (OUT/'media-metrics.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Qwen scaling review</title>
<style>body{font:16px system-ui;background:#161819;color:#eee;margin:2rem} .row{display:flex;gap:1rem}figure{margin:0;flex:1}video{width:100%;max-width:420px}button{padding:.7rem}figcaption{padding:.5rem}</style>
<h1>Qwen scaling comparison</h1><p>M4 Max · 320×320 · 124 frames · 20 steps · seed 72 · fresh Qwen in every render.</p>
<p>Clips start muted; unmute one clip to compare dialogue.</p><p>Numerical correctness does not establish universal generation-quality improvement. Compare subject identity, prompt adherence, motion and dialogue.</p>
<button onclick="document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play()})">Play all from start</button>
<button onclick="document.querySelectorAll('video').forEach(v=>v.pause())">Pause all</button>
'''+''.join(sections))
if __name__=='__main__':main()
