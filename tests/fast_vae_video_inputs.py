#!/usr/bin/env python3
"""Extend denoiser-free C4 sources to the released reference-video minimum."""
import hashlib, json, pathlib, subprocess

root=pathlib.Path('outputs/fast-vae/cuda')
out=root/'C5-video-inputs';out.mkdir(parents=True,exist_ok=True)
for name in ['faces','texture']:
    source=root/'pixels'/(name+'.mp4');target=out/(name+'.mp4')
    command=['ffmpeg','-v','error','-y','-i',str(source),'-vf',
        'tpad=stop_mode=clone:stop_duration=1','-frames:v','48','-r','24',
        '-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p',str(target)]
    subprocess.run(command,check=True)
    (out/(name+'.json')).write_text(json.dumps(dict(source=str(source),
        source_sha256=hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),
        frames=48,kind='reference-derived pan extended by last-frame hold; no denoising',
        command=command),indent=2)+'\n')
