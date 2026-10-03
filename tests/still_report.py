#!/usr/bin/env python3
"""Publish retained evidence; never runs a denoiser or changes acceptance gates."""
import hashlib,html,json,re,struct
from pathlib import Path
import numpy as np
from PIL import Image
R=Path('outputs/single-still')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8<<20):h.update(b)
 return h.hexdigest()
def metrics(a,b):
 d=a.astype(np.float64)-b;return {'mae':float(np.abs(d).mean()),'psnr':float(-10*np.log10(max(float(np.square(d).mean()),1e-30)))}
def document(title,body):return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+title+'</title><style>body{max-width:1500px;margin:32px auto;padding:0 24px;background:#f5f3ef;color:#212128;font:16px/1.5 system-ui}h1{font-size:32px}section{background:white;padding:20px;margin:24px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:18px}img{width:100%;height:auto;border-radius:6px}figure{margin:0}table{border-collapse:collapse}td,th{padding:8px 16px;text-align:left;border-bottom:1px solid #ddd}a{color:#175387}code{font-size:13px}small{color:#50505a}</style><body><h1>'+title+'</h1>'+body+'</body></html>'
def figure(path,label):return f'<figure><a href="{path}"><img loading="lazy" src="{path}" alt="{html.escape(label)}"></a><figcaption>{html.escape(label)}</figcaption></figure>'
def main():
 codec=json.loads((R/'codec-metrics.json').read_text());assert all(x['pass'] for x in codec)
 cuda_file=Path('outputs/single-still-cuda/results.json')
 cuda_report=json.loads(cuda_file.read_text()) if cuda_file.exists() else None
 cuda_qualified=bool(cuda_report and cuda_report.get('pass'))
 contract=json.loads(Path('tests/still_contract.json').read_text())
 scripts={sha(Path('tests/still_reference.py'))}
 scripts.update(sha(p) for p in (R/'source').glob('still-reference*.py'))
 body='<p>Actual T=1, F16 checkpoint expanded to F32. Unclamped RGB is compared before PNG rounding. Local M4 Max evidence.</p><p><a href="generation.html">Generation examples</a> · <a href="results.json">Full machine-readable results</a></p>'
 body+=('<p><a href="../single-still-cuda/review.html">Separate CUDA qualification and comparisons</a></p>' if cuda_qualified else '<p>CUDA is unqualified in this evidence set.</p>')
 controls={}
 for row in codec:
  w,h=row['size'];p=R/'fixtures'/str(w);source=np.fromfile(p/'source.f32',np.float32).reshape(h,w,3);ref=np.load(p/'reference_slices.npy');cand=np.fromfile(p/'candidate.f32',np.float32).reshape(h,w,3)
  provenance=json.loads((p/'reference.json').read_text())
  assert provenance['contract_sha256']==sha('tests/still_contract.json') and provenance['model_sha256']==contract['artifact_sha256']
  assert provenance['latent_sha256']==sha(p/'latent.safetensors') and provenance['script_sha256'] in scripts
  comparison=metrics(cand,ref[3]);assert comparison['mae']<=contract['gates']['unclamped_rgb_mae_max'] and comparison['psnr']>=contract['gates']['unclamped_rgb_psnr_min_db']
  assert (p/'output.png').exists();body+=f'<section><h2>{w}×{h}</h2><p>Candidate/reference PSNR {row["candidate_reference"]["psnr"]:.2f} dB; source/reconstruction {row["source_candidate"]["psnr"]:.2f} dB.</p><div class="grid">'
  for name,label in [('source','Source crop'),('reference','Independent F32 reference'),('output','h3cli reconstruction')]:body+=figure(f'fixtures/{w}/{name}.png',label)
  body+='</div><p><a href="fixtures/'+str(w)+'/metrics.json">Metrics</a> · <a href="fixtures/'+str(w)+'/decode.log">Load/reuse timing</a></p></section>'
  if w==256:
   controls['slices']=[]
   for i in range(4):
    a=np.fromfile(p/f'candidate-slice-{i}.f32',np.float32).reshape(h,w,3);m=metrics(a,ref[i]);assert m['mae']<=.01 and m['psnr']>=35;controls['slices'].append(m)
   wrong=np.fromfile(p/'wrong-normalization.f32',np.float32).reshape(h,w,3);controls['wrong_normalization']=metrics(wrong,cand)
   Image.fromarray(np.uint8(np.clip(wrong,0,1)*255+.5)).save(p/'wrong-normalization.png')
   body+='<section><h2>Slice and normalization controls</h2><div class="grid">'+''.join(figure(f'fixtures/256/slice-{i}.png',f'Reference slice {i}'+(' (declared output)' if i==3 else '')) for i in range(4))+figure('fixtures/256/wrong-normalization.png','Wrong normalization: raw supplied as normalized')+'</div><p>All four candidate slices match their corresponding independent reference. Only metadata slice 3 is used publicly.</p></section>'
  if w==512:
   whole=np.load(p/'whole_slices.npy')[3];controls['whole_vs_tiled']=metrics(whole,ref[3]);controls['whole_vs_source']=metrics(whole,source)
   Image.fromarray(np.uint8(np.clip(whole,0,1)*255+.5)).save(p/'whole.png')
   body+='<section><h2>Whole-image diagnostic</h2><div class="grid">'+figure('fixtures/512/reference.png','Released 256/64 tiles')+figure('fixtures/512/whole.png','Untiled diagnostic, different RoPE context')+'</div><p>Larger tiles are a different numerical recipe and remain unavailable in the public image decoder.</p></section>'
 video={}
 for n in [22,243]:
  a=json.loads((R/'video'/f'baseline-{n}.json').read_text());b=json.loads((R/'video'/f'current-{n}.json').read_text());assert a['frames']==b['frames']==n and a['rgb_f32_sha256']==b['rgb_f32_sha256'];video[str(n)]={'pass':True,'baseline':a,'current':b}
 body+='<section><h2>Video preservation</h2><p>Original and updated decoders produced identical F32 RGB checksums at 22 and 243 frames. The 243-frame test reused a retained latent state; it performed no denoising.</p></section>'
 (R/'review.html').write_text(document('Single-still codec qualification',body))
 generated={};g='<p>H3 single-image sampling: one video latent frame, two auxiliary audio ticks generated and discarded. Original BF16 weights, dense attention, 50 blocks, six Euler evaluations, 640×480, seed 42.</p><p><a href="review.html">Codec comparisons</a> · <a href="generation/determinism.json">Exact seed-repeat evidence</a></p>'
 for name,label in [('prompt','Prompt-only: red ceramic teapot'),('repeat','Same prompt and seed, repeated'),('reference','Image reference: portrait of the woman in Picture 1')]:
  p=R/'generation'/name;row=json.loads((p/'result.json').read_text());assert row['pass'];log=(p/'render.log').read_text();m=re.search(r'Euler denoise\s+wall=\s*([\d.]+)s',log);row['denoise_seconds']=float(m[1]) if m else None
  row['png_sha256']=sha(p/'output.png');row['latent_sha256']=sha(p/'latent.safetensors');generated[name]=row
  g+=f'<section><h2>{label}</h2><p>Wall {row["wall_seconds"]:.2f}s; denoising {row["denoise_seconds"]:.2f}s; sampled peak footprint {row["peak_footprint"]/1e9:.2f} GB.</p><div class="grid">'
  if name=='reference':g+=figure('../../inputs/2.jpg','Picture 1 input')
  g+=figure(f'generation/{name}/output.png',label)+'</div><p><a href="generation/'+name+'/render.log">Execution log</a> · <a href="generation/'+name+'/latent.safetensors">Normalized still latent</a></p></section>'
 assert generated['prompt']['latent_sha256']==generated['repeat']['latent_sha256']
 assert sha(R/'generation/prompt/output.png')==sha(R/'generation/prompt/redecoded.png')
 g+='<p>The output screens show the requested red teapot and a portrait consistent with the supplied image. These are bounded smoke examples, not a broad quality benchmark or 50-step qualification. Accelerated SOL/FP16/Q8/Turbo compositions are not qualified by these results.</p>'
 (R/'generation.html').write_text(document('Single-still generation examples',g))
 # Retain effective Euler velocity fixtures derived from authoritative transitions.
 sig=json.loads((R/'layout/references-0.json').read_text());dtv=np.diff(sig['video_sigmas']);dta=np.diff(sig['audio_sigmas'])
 for name in generated:
  p=R/'generation'/name;states=[np.fromfile(p/f'step-{i}.f32',np.float32) for i in range(7)]
  assert all(len(x)==24*30*40+128 and np.isfinite(x).all() for x in states)
  velocity=np.stack([np.concatenate([(states[i+1][:-128]-states[i][:-128])/dtv[i],(states[i+1][-128:]-states[i][-128:])/dta[i]]) for i in range(6)])
  np.save(p/'effective_euler_velocity.npy',velocity.astype(np.float32))
 cuda={'compiler_available':cuda_qualified,'device_available':cuda_qualified,'execution':'qualified default CUDA on RTX PRO 6000 Blackwell' if cuda_qualified else 'unqualified','portable_host_syntax':'passed'}
 if cuda_qualified:cuda['evidence']='../single-still-cuda/results.json'
 results={'codec':codec,'controls':controls,'video':video,'generation':generated,'cuda':cuda,'semantic_screen':'limited retained examples; no broad or 50-step claim'}
 (R/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 files=[p for p in R.rglob('*') if p.is_file() and 'sanitizers' not in p.parts and p.name not in ('checksums.json','report.log')]
 (R/'checksums.json').write_text(json.dumps({str(p.relative_to(R)):sha(p) for p in sorted(files)},indent=2)+'\n')
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
