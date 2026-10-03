#!/usr/bin/env python3
"""Build offline paired-media pages with optional human-review controls."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path, help='Qualification outputs containing held-out media and decoder replay directories')
    p.add_argument('--output', default='review.html')
    p.add_argument('--skip-human-review', action='store_true',
                   help='Provide playback pages without verdicts, notes or review-completion checks')
    a = p.parse_args()
    groups = {}
    paths = [*a.root.rglob('heldout-*.mp4'), *a.root.rglob('production-*.mp4')]
    for path in sorted(paths):
        match = re.fullmatch(r'heldout-(fl|ref)-(base|turbo)-(off|fp8|nvfp4)-(1200[123])-r0-(default|sage2\+\+|sage3)(?:-(preview|full)-(sage|non-sage))?\.mp4', path.name)
        if match:
            family, model, projection, seed, mode, vae, build = match.groups()
            repeat = None
        else:
            match = re.fullmatch(r'production-(fl|ref)-(turbo)-(off|nvfp4)-(42)-r([01])-(default|sage2\+\+|sage3)\.mp4', path.name)
            if not match:
                continue
            family, model, projection, seed, repeat, mode = match.groups()
            vae = build = None
        record=path.with_suffix('.json')
        if not record.exists() or json.loads(record.read_text()).get('returncode') != 0:
            continue
        if build == 'non-sage' or vae == 'preview':
            continue  # Native generation already provides the matching preview.
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_frames', '-of', 'json', str(path)]))['streams'][0]
        geometry = f'{probe["width"]}×{probe["height"]}/{probe["nb_frames"]} frames'
        label = f'{family.upper()}2VA {model}, {projection} projections, seed {seed}, {geometry}, {vae or "preview"} VAE'
        if repeat is not None:
            label = f'Target settings, pair {int(repeat) + 1}: {label}'
        entry = groups.setdefault(label, {'label': label, 'clips': {}})
        entry['clips'][mode] = str(path.relative_to(a.root))
    # A paired review requires all three modes, including a full-VAE baseline.
    complete = [r for r in groups.values() if set(r['clips']) == {'default', 'sage2++', 'sage3'}]
    for group in complete:
        group['clip_sha256'] = {mode:hashlib.sha256((a.root / path).read_bytes()).hexdigest()
                                for mode,path in group['clips'].items()}
        group['fingerprint'] = hashlib.sha256(json.dumps(group['clip_sha256'],sort_keys=True).encode()).hexdigest()
    if a.skip_human_review:
        for page in a.root.rglob('review.html'):
            old = page.read_text()
            updated = old.replace('Human playback/listening review: pending. Numeric checks do not establish perceptual quality.',
                                  'Recorded clips for playback.')
            if updated != old:
                page.write_text(updated)
    links = []
    for page in sorted(a.root.glob('continuation-*/review.html')):
        links.append({'name': page.parent.name, 'path': str(page.relative_to(a.root))})
    data = json.dumps({'groups': complete, 'continuations': links,
                      'human_review_skipped': a.skip_human_review}).replace('<', '\\u003c')
    review_controls = '''<p><label>Human review <select id="verdict"><option value="pending">Not reviewed</option>
<option value="acceptable">Acceptable differences</option><option value="visual">Visual issues</option>
<option value="audio">Audio issues</option><option value="both">Visual and audio issues</option></select></label></p>
<textarea id="notes" aria-label="Review notes" placeholder="Note the mode, time and observed issue; distinguish differences from degradation."></textarea>
<button id="download">Download review notes</button>''' if not a.skip_human_review else ''
    html = '''<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3 attention media review</title>
<style>
body{font:16px system-ui,sans-serif;margin:24px;background:#171717;color:#eee;line-height:1.5}
button,select,textarea{font:inherit;padding:6px;margin:4px 4px 4px 0}
button{cursor:pointer}a{color:#9cd4ff}.clips{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
video{display:block;width:100%;max-height:65vh;background:#000}.controls{margin:16px 0}
textarea{box-sizing:border-box;width:100%;min-height:90px}#seek{width:100%}
@media(max-width:700px){.clips{grid-template-columns:1fr}body{margin:12px}}
</style>
<h1>H3 attention media review</h1>
<p>Compare detail, identity, motion/flicker, prompt adherence, speech/voice,
distortion and lip sync. Clips in each group share model, projection precision,
prompt, seed and geometry. Numeric gates are reported independently.</p>
<div><button id="previous">Previous</button><select id="group"></select><button id="next">Next</button></div>
<div class="controls"><button id="play">Play all</button><button id="pause">Pause all</button>
<label>Listen to <select id="listen"><option>default</option><option>sage2++</option><option>sage3</option></select></label>
<input id="seek" type="range" min="0" max="1" step="0.01" value="0" aria-label="Seek all clips"></div>
<div class="clips" id="clips"></div>
REVIEW_CONTROLS
<span id="count"></span>
<h2>Continuation seams</h2><p>Open the segment galleries to inspect each boundary.</p><ul id="chains"></ul>
<script>
const evidence=DATA;
const reviewEnabled=!evidence.human_review_skipped;
const modes=['default','sage2++','sage3'];
const select=document.querySelector('#group'), clips=document.querySelector('#clips');
const verdict=document.querySelector('#verdict'), notes=document.querySelector('#notes');
const key='h3-attention-review-v1';let reviews={};
if(reviewEnabled){try{reviews=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}}
let current=null;
function videos(){return Array.from(clips.querySelectorAll('video'))}
function saved(g){const r=reviews[g.label];return r&&r.fingerprint===g.fingerprint?r:{verdict:'pending',notes:''}}
function save(){
 if(!reviewEnabled){document.querySelector('#count').textContent=evidence.groups.length+' comparison groups';return}
 if(current){reviews[current.label]={verdict:verdict.value,notes:notes.value,fingerprint:current.fingerprint};
  try{localStorage.setItem(key,JSON.stringify(reviews))}catch(e){}}
 document.querySelector('#count').textContent=' '+evidence.groups.filter(g=>saved(g).verdict!=='pending').length+' / '+evidence.groups.length+' groups reviewed';
}
function sound(){videos().forEach((v,i)=>v.muted=modes[i]!==document.querySelector('#listen').value)}
function show(){
 save();videos().forEach(v=>v.pause());clips.replaceChildren();
 current=evidence.groups[Number(select.value)];if(!current)return;
 modes.forEach(mode=>{const box=document.createElement('section'),label=document.createElement('h2'),v=document.createElement('video');
  label.textContent=mode;v.src=current.clips[mode];v.controls=true;v.preload='metadata';v.muted=true;
  v.addEventListener('loadedmetadata',()=>document.querySelector('#seek').max=v.duration);
  box.append(label,v);clips.append(box)});
 if(reviewEnabled){const r=saved(current);verdict.value=r.verdict;notes.value=r.notes;}
 document.querySelector('#seek').value=0;sound();save();
 videos()[0].addEventListener('timeupdate',()=>document.querySelector('#seek').value=videos()[0].currentTime);
}
evidence.groups.forEach((g,i)=>{const o=document.createElement('option');o.value=i;o.textContent=g.label;select.append(o)});
evidence.continuations.forEach(g=>{const li=document.createElement('li'),a=document.createElement('a');a.href=g.path;a.textContent=g.name;li.append(a);document.querySelector('#chains').append(li)});
select.addEventListener('change',show);
document.querySelector('#previous').onclick=()=>{select.selectedIndex=Math.max(0,select.selectedIndex-1);show()};
document.querySelector('#next').onclick=()=>{select.selectedIndex=Math.min(evidence.groups.length-1,select.selectedIndex+1);show()};
document.querySelector('#play').onclick=()=>videos().forEach(v=>v.play().catch(()=>{}));
document.querySelector('#pause').onclick=()=>videos().forEach(v=>v.pause());
document.querySelector('#listen').onchange=sound;
document.querySelector('#seek').oninput=e=>videos().forEach(v=>v.currentTime=Number(e.target.value));
if(reviewEnabled){verdict.onchange=save;notes.oninput=save;
document.querySelector('#download').onclick=()=>{save();const rows=evidence.groups.map(g=>({...g,...saved(g)}));
 const url=URL.createObjectURL(new Blob([JSON.stringify({schema:1,source:'human playback notes',recorded_at:new Date().toISOString(),groups:rows},null,2)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='attention-human-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};}
if(evidence.groups.length)show();else document.querySelector('#count').textContent='No complete held-out pairs copied yet.';
</script>'''.replace('DATA', data).replace('REVIEW_CONTROLS', review_controls)
    destination = a.root / a.output
    destination.write_text(html)
    print(f'{destination}: {len(complete)} paired groups, {len(links)} continuation galleries')


if __name__ == '__main__':
    main()
