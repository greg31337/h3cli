#!/usr/bin/env python3
"""Publish completed CUDA qualification without running GPU work."""
import hashlib
import html
import json
from pathlib import Path
import re
import numpy as np

ROOT=Path('outputs/single-still-cuda')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8<<20):h.update(b)
    return h.hexdigest()

def fig(path,label):
    return f'<figure><a href="{path}"><img loading="lazy" src="{path}" alt="{html.escape(label)}"></a><figcaption>{html.escape(label)}</figcaption></figure>'

def main():
    codec=json.loads((ROOT/'codec.json').read_text())
    video=json.loads((ROOT/'video.json').read_text())
    generation=json.loads((ROOT/'generation.json').read_text())
    deterministic=json.loads((ROOT/'determinism.json').read_text())
    slices=json.loads((ROOT/'slice-controls.json').read_text())
    assert len(codec)==3 and all(r['pass'] for r in codec)
    assert all(video[str(n)]['pass'] for n in [22,243])
    assert set(generation)=={'prompt','repeat','reference'} and all(r['pass'] for r in generation.values())
    assert deterministic['pass'] and len(deterministic['initial_and_step_sha256'])==7
    for i in range(7):
        for name in ['prompt','repeat']:
            assert sha(ROOT/f'generation/{name}/step-{i}.f32')==deterministic['initial_and_step_sha256'][i]
    assert all(r['mae']<=.01 and r['psnr']>=35 for r in slices)
    proper=np.fromfile(ROOT/'fixtures/256/candidate.f32',np.float32)
    wrong=np.fromfile(ROOT/'fixtures/256/wrong-normalization.f32',np.float32)
    assert proper.shape==wrong.shape and np.isfinite(wrong).all() and not np.array_equal(proper,wrong)
    delta=proper.astype(np.float64)-wrong
    normalization={'mae':float(np.abs(delta).mean()),'psnr':float(-10*np.log10(np.square(delta).mean()))}
    assert 'ERROR SUMMARY: 0 errors' in (ROOT/'gpu-memcheck.log').read_text()
    assert 'PASS' in (ROOT/'lifecycle.log').read_text()
    assert 'atomic PNG failure passed' in (ROOT/'controls.log').read_text()
    assert sha(ROOT/'generation/prompt/output.png')==sha(ROOT/'generation/prompt/redecoded.png')
    for name,row in generation.items():
        log=(ROOT/f'generation/{name}/render.log').read_text()
        # CUDA profile marks are cumulative since context creation. Use the
        # explicit denoise progress interval, not the cumulative "Euler" mark.
        row.setdefault('denoise_seconds',None)
        phase=re.search(r'still cold_load=([0-9.]+)s decode=([0-9.]+)s delivery=([0-9.]+)s',log)
        assert phase
        row.update(image_load_seconds=float(phase[1]),image_decode_seconds=float(phase[2]),delivery_seconds=float(phase[3]))
        peaks=re.findall(r'peak(?:=\s*| )([0-9.]+)\s*GiB',log)
        row['reported_peak_gpu_gib']=max(map(float,peaks)) if peaks else None
        assert row['png_sha256']==sha(ROOT/f'generation/{name}/output.png')
        assert row['latent_sha256']==sha(ROOT/f'generation/{name}/latent.safetensors')
    result={'pass':True,'device':'RTX PRO 6000 Blackwell Server Edition, SM120, 96 GB',
            'cuda_toolkit':'12.8','driver':'595.91.07','codec':codec,'video':video,
            'generation':generation,'determinism':deterministic,'slice_controls':slices,'wrong_normalization':normalization,
            'scope':'Default CUDA F32 VAE, dense BF16 DiT; no Sage/NVFP4/fast-CUDA qualification',
            'memory_semantics':'peak_footprint is sampled Linux RSS + swap; GPU peak is reported separately',
            'quality_scope':'Bounded six-step examples, not broad semantic or 50-step qualification'}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    body='<p>RTX PRO 6000 Blackwell (96 GB), CUDA 12.8. Default CUDA execution; F32 image VAE, dense BF16 DiT. All GPU jobs ran serially.</p><p><a href="results.json">Results</a> · <a href="commands.jsonl">Exact commands</a> · <a href="gpu-memcheck.log">Compute Sanitizer</a></p>'
    if (ROOT.parent/'single-still/review.html').exists():body+='<p><a href="../single-still/review.html">Metal codec evidence</a></p>'
    for row in codec:
        w,h=row['size'];m=row['same_latent_reference'];cross=row['same_latent_metal']
        body+=f'<section><h2>{w}×{h} codec</h2><p>Same-latent CUDA/reference PSNR {m["psnr"]:.2f} dB, MAE {m["mae"]:.3g}; CUDA/Metal PSNR {cross["psnr"]:.2f} dB.</p><div class="grid">'
        body+=fig(f'fixtures/{w}/source.png','Original source crop')+fig(f'fixtures/{w}/reference.png','Independent F32 reference')+fig(f'fixtures/{w}/output.png','CUDA reconstruction')+'</div>'
        body+=f'<p>Fresh CUDA encoder/decoder round trip: source PSNR {row["source_candidate"]["psnr"]:.2f} dB; independent CUDA F32 decoder PSNR {row["roundtrip_reference"]["psnr"]:.2f} dB.</p></section>'
    body+='<section><h2>Video preservation and lifecycle</h2><p>Original and updated CUDA video decoders have identical F32 RGB checksums at 22 and 243 frames. These tests reused a retained latent state and performed no video denoising. All four image slices, cancellation/retry, constant repeated allocation and atomic PNG failure checks passed.</p></section>'
    for name,title in [('prompt','Prompt-only: red ceramic teapot'),('repeat','Repeated prompt and seed'),('reference','Image-reference portrait')]:
        row=generation[name];d=row['denoise_seconds'];dt=f'{d:.2f} s' if d is not None else 'not separately recorded'
        body+=f'<section><h2>{title}</h2><p>{html.escape(row["prompt"])}</p><p>640×480, six evaluations, seed 42. Wall {row["wall_seconds"]:.2f} s; denoising {dt}; sampled host footprint {row["peak_footprint"]/1e9:.2f} GB.</p><div class="grid">'
        if name=='reference':body+=fig('../../inputs/2.jpg','Picture 1')
        body+=fig(f'generation/{name}/output.png','CUDA output')
        if name!='repeat' and (ROOT.parent/f'single-still/generation/{name}/output.png').exists():body+=fig(f'../single-still/generation/{name}/output.png','Metal output with the same prompt and seed')
        body+='</div><p><a href="generation/'+name+'/render.log">Execution log</a></p></section>'
    body+='<p>CUDA repeats match every retained state and the final PNG exactly. Independent backends need not produce identical diffusion trajectories. The examples do not qualify 50-step quality or accelerated precision/routing compositions. The first generation used the original test harness; repeat/reference cases add explicit denoising-interval timing without changing inference. Cumulative CUDA profile marks are not reported as denoising duration.</p>'
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>CUDA single-still qualification</title><style>body{max-width:1500px;margin:32px auto;padding:0 24px;background:#f5f3ef;color:#212128;font:16px/1.5 system-ui}section{background:white;padding:20px;margin:24px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:18px}img{width:100%;height:auto;border-radius:6px}figure{margin:0}a{color:#175387}</style><body><h1>CUDA single-still qualification</h1>'+body+'</body></html>'
    (ROOT/'review.html').write_text(page)
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name not in ('checksums.json','report.log')]
    (ROOT/'checksums.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},indent=2)+'\n')
    print('PASS: CUDA codec, lifecycle, video, generation and report evidence')

if __name__=='__main__':main()
