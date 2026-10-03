"""Read-only historical metrics; rendering uses cuda_single_run.py."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def inspect(path, width, height, frames):
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))
    v=next(x for x in info['streams'] if x['codec_type']=='video')
    a=next(x for x in info['streams'] if x['codec_type']=='audio')
    assert (v['width'],v['height'],int(v['nb_frames']))==(width,height,frames)
    assert v['r_frame_rate']=='24/1' and int(a['sample_rate'])==32000 and a['channels']==2
    assert abs(float(v['duration'])-float(a['duration']))<.1
    subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True,timeout=30)
    return dict(width=width,height=height,frames=frames,fps=24,audio_rate=32000,audio_channels=2,duration=float(v['duration']))

def gallery(out):
    records=[];cards=[]
    for manifest in sorted((out/'quality').glob('*.json')):
        if manifest.name.startswith('.'):continue
        row=json.loads(manifest.read_text())
        if not isinstance(row,dict) or row.get('role')!='held-out':continue
        records.append(row)
        cells=[]
        for mode in ('off','fp8','nvfp4'):
            artifact=row['outputs'].get(mode)
            if artifact:
                filename=Path(artifact['path']).name
                poster=Path(filename).with_suffix('.jpg')
                thumbnail=f'<img style="width:100%" src="{html.escape(str(poster))}" alt="First, middle and last frame">' if (out/'quality'/poster).exists() else ''
                cells.append(f'<div><b>{mode}</b><video controls preload="metadata" src="{html.escape(filename)}"></video>{thumbnail}</div>')
            else:cells.append(f'<div>{mode}: pending</div>')
        cards.append(f'<section><h2>{html.escape(row["name"])}</h2><p>{html.escape(row["prompt"])}</p><div class="row">'+''.join(cells)+'</div></section>')
    manifest_text=json.dumps(records,indent=2)+'\n'
    decisions=[]
    acceptance_path=out/'quality/acceptance.json'
    acceptance=json.loads(acceptance_path.read_text()) if acceptance_path.exists() else {}
    manifest_matches=acceptance.get('gallery_manifest_sha256')==hashlib.sha256(manifest_text.encode()).hexdigest()
    media_matches=all('off' in row['outputs'] and all(
        (out/'quality'/Path(entry['path']).name).is_file() and
        sha(out/'quality'/Path(entry['path']).name)==entry['sha256']
        for entry in row['outputs'].values()) for row in records)
    for mode in ('fp8','nvfp4'):
        decision=acceptance.get('decisions',{}).get(mode,{})
        expected={row['name']:row['outputs'][mode]['sha256'] for row in records if mode in row['outputs']}
        accepted=(manifest_matches and media_matches and bool(expected) and len(expected)==len(records)
                  and decision.get('status')=='accepted' and decision.get('media')==expected)
        decisions.append(f'{mode.upper()}: '+('human playback/listening accepted for these exact media.' if accepted else 'human playback/listening acceptance pending or stale.'))
    status=' '.join(decisions)
    (out/'quality/review.html').write_text('<!doctype html><meta charset="utf-8"><title>5090 denoiser quantization</title><style>body{background:#151515;color:#eee;font:16px sans-serif;margin:24px}video{display:block;width:100%;max-height:480px}.row{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}section{margin-bottom:40px}a{color:#8cf}</style><h1>BF16 / FP8 / NVFP4</h1><p>'+status+' Same noise/conditioning/sampler, full VAE, and fast-CUDA attention within each triplet.</p><p><a href="manifest.json">Hashes and commands</a> · <a href="acceptance.json">Quality decisions and scope</a> · <a href="../validation-index.json">Build/model identities and timings</a></p><p>The <a href="../followup/review.html">separate continuation and Turbo gallery</a> records its own review status. This base-model approval does not cover those outputs.</p>'+''.join(cards))
    (out/'quality/manifest.json').write_text(manifest_text)
