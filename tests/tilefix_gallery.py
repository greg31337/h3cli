#!/usr/bin/env python3
"""Build a local, synchronized video comparison page from matrix artifacts."""
import json
from pathlib import Path
from tilefix_validation import ROOT
out=ROOT/'outputs/tilefix-validation/matrix'
cases={d.name:json.loads((d/'runs.json').read_text()) for d in out.iterdir() if d.is_dir() and (d/'official.mp4').exists()}
html='''<!doctype html><meta charset="utf-8"><title>VideoVAE tile comparisons</title>
<style>body{background:#16191e;color:#eee;font:16px system-ui;margin:28px}button,select,input{font:inherit;margin:6px}section{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}video{width:100%;background:#000}h2{font-size:18px}a{color:#9bcaff}table{border-collapse:collapse}td,th{padding:8px;text-align:right;border-bottom:1px solid #555}p{max-width:1000px}</style>
<h1>VideoVAE tile comparisons</h1><p>Every view uses exactly the same saved latent for that canvas. The source pans across face, body and 2.jpg images, a gray gradient, and low-contrast lines. These are VAE reconstructions of encoded input, not DiT generations. Each clip has 22 frames at 24 fps.</p>
<label>Canvas (height × width) <select id="size"></select></label><button id="play">Play / pause all</button><button id="reset">Restart</button>
<label>Frame <input type="range" min="0" max="21" value="0" id="frame"><output id="number">0</output></label>
<section id="views"></section><p><a id="sheet">Frame comparison</a> · <a id="metrics">Pixel and seam metrics</a> · <a id="raw">Timings, memory and hashes</a></p><table id="stats"></table>
<p>Times exclude weight loading. RSS and Metal tensor memory overlap and must not be added. The official MPS memory sample is not a peak measurement. Larger tiles are unqualified; their reconstruction differs from the released 256/64 reference.</p>
<script>const data=DATA;const modes=['official','default','auto','320'];let videos=[];
const selector=document.getElementById('size'),frame=document.getElementById('frame');
Object.keys(data).sort().forEach(s=>selector.add(new Option(s.replace('x',' × '),s)));
function load(){const s=selector.value,r=data[s];document.getElementById('views').innerHTML=modes.map(m=>`<div><h2>${m==='official'?'Official 256':m==='default'?'Default 256':m==='auto'?'Legacy auto ('+r.auto.tile+')':'Explicit 320'}</h2><video muted playsinline loop preload="auto" src="${s}/${m}.mp4"></video></div>`).join('');videos=[...document.querySelectorAll('video')];
for(const [id,file] of [['sheet','comparison.png'],['metrics','metrics.json'],['raw','runs.json']])document.getElementById(id).href=s+'/'+file;
document.getElementById('stats').innerHTML='<tr><th>Mode</th><th>Spatial tiles</th><th>Decode seconds</th><th>Peak RSS GiB</th><th>Peak Metal tensors GiB</th></tr>'+modes.map(m=>`<tr><td>${m}</td><td>${r[m].spatial_tiles??'—'}</td><td>${r[m].decode_seconds.toFixed(2)}</td><td>${(r[m].peak_rss_bytes/2**30).toFixed(2)}</td><td>${r[m].peak_metal_tensor_bytes?(r[m].peak_metal_tensor_bytes/2**30).toFixed(2):'not sampled'}</td></tr>`).join('');frame.value=0;document.getElementById('number').value=0;}
selector.onchange=load;document.getElementById('play').onclick=()=>{if(videos[0].paused){let t=videos[0].currentTime;videos.forEach(v=>{v.currentTime=t;v.play()})}else videos.forEach(v=>v.pause())};
document.getElementById('reset').onclick=()=>videos.forEach(v=>{v.currentTime=0});
frame.oninput=()=>{videos.forEach(v=>{v.pause();v.currentTime=Number(frame.value)/24+.001});document.getElementById('number').value=frame.value};load();</script>'''.replace('DATA',json.dumps(cases))
(out/'index.html').write_text(html)
print(out/'index.html')
