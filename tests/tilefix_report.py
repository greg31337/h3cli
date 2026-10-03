#!/usr/bin/env python3
"""Collect completed tilefix evidence into a tracked machine-readable report."""
import hashlib,json
from pathlib import Path
from tilefix_validation import ROOT,MATRIX,sha
out=ROOT/'outputs/tilefix-validation'
matrix={}
for h,w in MATRIX:
    name=f'{h}x{w}';d=out/'matrix'/name
    runs=json.loads((d/'runs.json').read_text());metrics=json.loads((d/'metrics.json').read_text())
    assert all(k in runs for k in ('default','auto','320','official'))
    assert len({runs[k]['latent_sha256'] for k in ('default','auto','320','official')})==1
    comparisons=metrics['comparisons']
    for mode in ('auto','320'):
        assert comparisons['default-vs-official']['rmse']<=comparisons[mode+'-vs-official']['rmse']
        assert comparisons['default-vs-official']['ssim_luma_11x11']>=comparisons[mode+'-vs-official']['ssim_luma_11x11']
    matrix[name]={'runs':runs,'metrics':metrics}
report={'baseline_commit':'59b64c9','hardware':'Apple M4 Max, 128 GiB unified memory',
    'fixture':'22-frame deterministic pans, released Ref2VA encoder, shared normalized F32 latent per canvas',
    'inputs_sha256':{name:sha(ROOT/'inputs'/name) for name in ('face1.jpg','body1.jpg','2.jpg')},
    'source_audit':json.loads((out/'source-audit.json').read_text()),'matrix':matrix,
    'generation':json.loads((out/'generation/results.json').read_text()),
    'saved_state':json.loads((out/'saved/results.json').read_text()),
    'legacy_and_path_parity':json.loads((out/'compatibility/results.json').read_text()),
    'checks':{'configuration_and_overlap':25689,'configuration_sanitizers':'ASan/UBSan passed',
        'logging_unittests':2,'metric_unittests':3,
        'full_suite_passed':True,'optional_fixture_groups_skipped':10},
    'limitations':['Single timing sample per policy/canvas; resident loading is measured separately.',
        'Tracked Metal tensors exclude driver scratch; RSS and Metal share physical memory.',
        'Official decoder uses upstream Diffusers F32/MPS and released weights, not CUDA.',
        'Quality fixtures are encoded-input reconstructions. Generation regression uses four steps/all 50 DiT layers.',
        'Seam ratios are content-dependent diagnostics without pass/fail thresholds.']}
log=(out/'full-suite.log').read_text();assert 'skip:' in log
report['checks']['optional_fixture_groups_skipped']=log.count('skip:')
(ROOT/'docs/tilefix-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('Canvas | auto | tiles 256/auto/320 | seconds 256/auto/320 | PSNR 256/auto/320 | peak Metal GiB')
for name,item in matrix.items():
    r=item['runs'];m=item['metrics']['comparisons']
    print(name,r['auto']['tile'],[r[k]['spatial_tiles'] for k in ('default','auto','320')],
          [round(r[k]['decode_seconds'],2) for k in ('default','auto','320')],
          [round(m[k+'-vs-official']['psnr_db'],2) for k in ('default','auto','320')],
          [round(r[k]['peak_metal_tensor_bytes']/1024**3,3) for k in ('default','auto','320')])

rows=[]
for name,item in matrix.items():
    r=item['runs'];m=item['metrics']['comparisons'];modes=('default','auto','320')
    rows.append('| '+name.replace('x','×')+' | '+str(r['auto']['tile'])+' | '+
        '/'.join(str(r[k]['spatial_tiles']) for k in modes)+' | '+
        ' / '.join(f'{r[k]["decode_seconds"]:.2f}' for k in modes)+' | '+
        f'{100*(r["default"]["decode_seconds"]/r["auto"]["decode_seconds"]-1):.1f}%'+' | '+
        ' / '.join(f'{m[k+"-vs-official"]["psnr_db"]:.2f}' for k in modes)+' |')
generation=report['generation']
replayed=[case for case in ('fl2va','ref2va','continuation') if generation[case+'-after']['text_replayed']]
for case in ('fl2va','ref2va','continuation'):
    assert generation[case]['encoded_audio_exact'] and generation[case]['decoded_rgb_changed']
qwen=('Fresh Qwen text matched in all three pairs; no text replay was needed.' if not replayed else
      'Fresh Qwen text varied in '+', '.join(replayed)+'. Their original attempts are preserved in the `*-after-fresh/` directories. Those pairs were rerun with only baseline text replayed in isolated test builds. Reference pixels and VAE conditions were freshly computed on both sides. This isolates the pre-existing cold-Qwen variability documented in [memory validation](memory-validation.md); it does not claim fresh cold-Qwen bitwise determinism. No replay hooks exist in production.')
md='''# VideoVAE tile-default validation

All 45 tasks were implemented and validated on the **M4 Max, 128 GiB**. Fixed
256-pixel decoding is much closer to the official 256/64 reference across every
required canvas. The old optimizer remains available as opt-in `auto`;
numeric overrides above 256 also remain opt-in. Both have measured speed
benefits and observable reconstruction differences, so neither is presented as
an equivalent-quality production default.

[Policy and upgrade notes](tilefix.md) · [complete measurements and hashes](tilefix-validation.json) ·
[local synchronized video gallery](../outputs/tilefix-validation/matrix/index.html)

## Fixed-latent comparison

Seven deterministic 22-frame pans contain `face1.jpg`, `body1.jpg`, `2.jpg`, a
smooth gray gradient, and low-contrast lines. Each canvas was encoded once using
the unchanged released Ref2VA encoder. The saved `[24,7,H/16,W/16]` normalized F32
latent, verified by SHA-256, was consumed unchanged by native default, `auto`,
explicit 320, and official 256 decoders. These fixtures test reconstruction of
real image content and motion; they are not DiT quality generations.

The reference is upstream Diffusers `AutoencoderKLMiniMaxH3` 0.40.0 with the
released FL2VA VideoVAE weights, 256/64 tiling, and PyTorch 2.14.0 F32/MPS. The
converter, decoder source, and actual decoder-weight hashes are recorded. This
is an official-class M4 reference, not a CUDA benchmark. Both outputs receive
matching RGB denormalization/clamping before comparison.

The table orders triples as **256 / auto / 320**. Dimensions are **height × width**.
Times measure resident decode only; loading is reported separately in JSON.

| Canvas | Old/auto tile | Spatial tiles | Decode seconds | Default cost vs auto | PSNR vs official, dB |
| --- | ---: | --- | --- | ---: | --- |
'''+ '\n'.join(rows)+'''

Across the seven cases, default mean absolute RGB error is 2.47e-6–5.94e-6 and
luminance SSIM is 0.99999937–0.99999982. Native and official output are **not
bit-identical**: maximum single-channel pixel errors range from 0.0319 to 0.0826.
PSNR, SSIM, MAE, RMSE, and maximum error for every pair are preserved. Explicit
320 is byte-identical to `auto` wherever `auto` selects 320.

Each timing is one measured pass in a separate process; no concurrent GPU jobs
ran. No claim about statistical confidence or full-quality generation speedup
is made. The largest canvas completed all four 22-frame decodes with finite
output. At 320×320, the new default uses four overlapping tiles instead of one;
its stronger official parity comes with the largest proportional decode cost.

## Memory and seams

Tracked peak Metal tensor memory was 9.365 GiB for 256 at every canvas, versus
9.408/9.454/9.503/9.554 GiB for 272/288/304/320. Native maximum process RSS was
9.24–10.14 GiB at 256 and 9.20–10.09 GiB under `auto`. The largest measured RSS
increase versus `auto` was about 94 MiB, consistent with retaining more
intermediate RGB tiles. There was no large memory regression. RSS and Metal
measurements overlap; tensor accounting excludes driver scratch. Official MPS
memory values are completion samples, not peak measurements.

Visual review covered the first, middle, and last frames of every canvas, full
resolution middle-frame exports, temporal mean error maps, and all-frame temporal
slices of the flat band during the pan. Default and official views were visually
indistinguishable at contact-sheet scale. Larger tiles changed facial detail,
hair, fabric texture, line contrast, and low-contrast shading. The amplified
576×1024 temporal slices show these differences persisting across the pan;
default's slice is nearly neutral. The native clips remained coherent; this
comparison does not establish that 256 eliminates every perceptual seam or that
larger tiles always look worse to a viewer.

The seam utility measures both overlap edges on each axis, using boundary-to-
nearby ratios for pixel gradients and four-pixel luminance steps. At 576×1024,
mean horizontal/vertical gradient ratios are 0.950/1.039 for 256 versus
1.011/1.113 for `auto` and 320. Results are not monotonic across all canvases:
at 512×512, a 256-policy boundary coincides with the deliberately sharp
photo-to-gradient source edge, inflating its horizontal ratio to 5.759. That is
why seam metrics remain diagnostics, with no invented pass/fail threshold.

## Generation, continuation, and saved state

Pinned baseline `59b64c9` and current builds ran FL2VA, Ref2VA, and a 39-frame
protected continuation at 320×320, seed 72, 56 target frames, four denoising
steps, and all 50 DiT blocks. The regression captures reference pixels, VAE
conditioning, token IDs/positions/spans, text, final video/audio latents, decoded
RGB, and PCM. Final latents, conditioning, PCM, saved `.h3av`, and demuxed AAC are
byte-identical with identical text; video RGB changes as expected. The
continuation pair uses the same complete baseline AV state and returns the
expected 17 delivered frames after trimming protected history.

'''+qwen+'''

Previously saved `.h3av` and current-schema `.h3sample` artifacts were loaded
through the unchanged native loaders and decoded in default, 256, `auto`, and
320 modes. A 320×320 AV state from the old-policy generation reproduces the old
RGB with `auto` and the new RGB with default/256. The existing 256×256 artifacts
produce identical pixels in all four policies because their actual tile extent
stays 256. No format conversion or version bump was needed. Full sampler resume
still enforces its existing build/environment restrictions; loading its latent
for reconstruction does not bypass exact-resume checks.

A pinned old decoder also reproduced the new `auto` result byte-for-byte at
576×1024. Standalone default and explicit 256 matched the resident default at
320×320. The old selector's scoring body is byte-identical, and all decoder code
outside the policy helpers is byte-identical to the baseline. No encoder,
reference-video, Qwen, DiT, audio, continuation, or serialization implementation
changed.

## Test evidence and task coverage

- 25,689 policy/geometry assertions passed normally and under ASan/UBSan,
  including every public supported canvas, all aligned numeric values through
  512, captured legacy selections, invalid/empty fallback, and unchanged encoder
  versus decoder 256/64 spatial plans.
- Two logging tests cover six policies plus quiet mode; three analytic metric
  tests cover identity, a known luminance shift, and an artificial seam.
- `make -j8 test` passed. Ten pre-existing optional fixture groups were skipped
  because their fixtures are absent; the real-model matrix above ran separately.
- Fixed-latent data, F32 outputs, all 28 MP4s, timings, memory, logs, frame exports,
  and checksums are in `outputs/tilefix-validation/`. The scripts and pinned
  Python requirements are in `tests/tilefix*` and `tests/requirements-tilefix.txt`.

| Tasks | Evidence |
| --- | --- |
| T001–T009 | Decoder helper/parser/logging; unchanged overlap and encoder code |
| T010–T015 | Configuration, legacy matrix, invalid input, logging and sanitizer tests |
| T016–T023 | Decode-only tool; seven immutable saved latents; 28 native/reference decodes and pixel metrics |
| T024–T027 | Seam analysis; all-canvas frame review, pan temporal slices, MP4 gallery |
| T028–T031 | 320×320 and 768×1344 stability; measured decoder timing and memory matrix |
| T032–T036 | FL2VA, Ref2VA, continuation, final-latent/conditioning/PCM/AAC comparisons with the Qwen qualification above |
| T037 | Existing AV/sampler artifact loading and four-policy reconstruction |
| T038–T043 | README, public header, policy guide and upgrade notes |
| T044–T045 | Preserved exact legacy heuristic; old/new pixel comparison; evidence-based tile-policy decision |
'''
(ROOT/'docs/tilefix-validation.md').write_text(md)
