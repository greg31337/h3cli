#!/usr/bin/env python3
"""Validate bridge integration artifacts and produce a small render review page.

Requires numpy and Pillow (the existing continuation-validation environment has
both). Quality comparisons and parameter sweeps belong to T043 onward.
"""
import argparse
import hashlib
import html
import json
import re
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

REQUIRED = {'bridge-unchanged', 'bridge-pose', 'bridge-to-bridge', 'bridge-to-hard',
            'image2', 'image-video', 'image-audio', 'video-audio', 'replaced-audio',
            't2va', 'trim-normal', 'zero-hard'}
ROW = re.compile(r'bridge step (\d+) class (\d+) mask (\S+) raw_rms (\S+) scaled_rms (\S+) update_rms (\S+) changed (\d+)/(\d+)')


def state(path):
    data = path.read_bytes()
    assert data[:8] == b'H3AV\r\n\x1a\n'
    assert struct.unpack_from('<4I', data, 8) == (3, 160, 0x01020304, 1)
    width, height, frames, vt, lh, lw, at, vc, ac, stereo = struct.unpack_from('<10I', data, 24)
    assert (vc, ac, stereo) == (24, 32, 2)
    nv, na = 24 * vt * lh * lw, 64 * at
    assert struct.unpack_from('<2Q', data, 72) == (nv*4, na*4)
    assert len(data) == 160 + (nv + na)*4
    assert hashlib.sha256(data[:128] + data[160:]).digest() == data[128:160]
    values = np.frombuffer(data, dtype='<f4', offset=160)
    assert np.isfinite(values).all()
    return {'data': data, 'width': width, 'height': height, 'frames': frames,
            'video': values[:nv].reshape(24, vt, lh, lw), 'audio': values[nv:].reshape(32, 2, at)}


def diagnostics(text, steps, passes=1):
    matches = ROW.findall(text)
    exact = re.findall(r'bridge step (\d+) exact video/audio bits unchanged', text)
    assert len(exact) == steps * passes
    counts = [0] * steps
    largest_scale_error = 0.
    for step, kind, strength, raw, scaled, update, changed, elements in matches:
        step, kind = int(step), int(kind)
        strength, raw, scaled, update = map(float, [strength, raw, scaled, update])
        changed, elements = int(changed), int(elements)
        assert 1 <= step <= steps and elements > 0 and 0 <= changed <= elements
        counts[step-1] += 1
        if strength == 0:
            assert scaled == update == changed == 0
        else:
            assert raw > 0 and update > 0 and changed > 0
            error = abs(scaled / raw - strength)
            largest_scale_error = max(largest_scale_error, error)
            assert error < 2e-6, (step, kind, strength, error)
        if strength == 1:
            assert scaled == raw, 'generated rows must retain unscaled velocity'
    assert min(counts) >= 4 * passes
    return {'transitions': len(exact), 'class_measurements': len(matches), 'max_velocity_scale_error': largest_scale_error}


def profile_metrics(text, passes, saved):
    packed = re.findall(r'bridge class (\d+) mask (\S+) packed rows (\d+)', text)
    video_rows = audio_rows = 0
    for kind, strength, count in packed:
        kind, count = int(kind), int(count)
        assert 0 <= float(strength) <= 1 and count > 0
        if kind in (0,2) or 4 <= kind < 14:
            video_rows += count
        else:
            assert kind in (1,3) or 14 <= kind < 24
            assert count % 2 == 0, 'both stereo planes must be classified'
            audio_rows += count
    _, vt, lh, lw = saved['video'].shape
    assert video_rows == passes * vt * lh * lw // 4
    assert audio_rows == passes * 2 * saved['audio'].shape[2]
    construction = re.findall(r'bridge profile construction (\S+) s, (\d+) bytes',text)
    initialization = re.findall(r'bridge AV initialization (\S+) s;',text)
    preparation = re.findall(r'bridge modulation preparation (\S+) s; (\d+) time rows \((\d+) extra\), (\d+) extra AdaLN bytes, (\d+) class-plan bytes',text)
    snapshots = re.findall(r'bridge exact-row audit snapshot: (\d+) bytes',text)
    maps = re.findall(r'bridge modulation maps: (\d+) bytes \((\d+) extra versus hard\)',text)
    assert len(construction) == len(initialization) == len(snapshots) == len(maps) == passes
    assert preparation, 'at least one modulation preparation, with cached passes allowed'
    assert all(int(extra) == 0 for _,extra in maps)
    assert all(int(size) == saved['video'].nbytes + saved['audio'].nbytes for size in snapshots)
    assert all(float(t) >= 0 for t,_ in construction) and all(float(t) >= 0 for t in initialization)
    return {'packed_video_rows_per_pass': video_rows // passes, 'packed_audio_rows_per_pass': audio_rows // passes,
            'construction_seconds': [float(t) for t,_ in construction],
            'initialization_seconds': [float(t) for t in initialization],
            'modulation_preparations': [{'seconds':float(t),'time_rows':int(rows),'extra_rows':int(extra),
                'extra_adaln_bytes':int(memory),'class_plan_bytes':int(plan)} for t,rows,extra,memory,plan in preparation],
            'exact_snapshot_bytes':int(snapshots[0]), 'extra_packed_map_bytes':0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=Path('outputs/bridge-integration/acceptance'))
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args(); out = args.directory.resolve()
    records = json.loads((out/'runs.json').read_text())
    if args.require_complete:
        assert REQUIRED <= records.keys(), sorted(REQUIRED-records.keys())
    result = {'passed': False, 'complete': REQUIRED <= records.keys(), 'renders': {}, 'diagnostics': {}, 'profiles': {}}
    review = out/'review'; review.mkdir(exist_ok=True)
    cards = []
    states, pixels, audio = {}, {}, {}
    for name, record in records.items():
        assert record['returncode'] == 0, name
        log = (out/(name+'.log')).read_text()
        bridge_cases = []
        for case, hashes in record['outputs'].items():
            for extension, expected in hashes.items():
                assert hashlib.sha256((out/(case+'.'+extension)).read_bytes()).hexdigest() == expected, (case, extension)
            meta = json.loads((out/(case+'.json')).read_text())
            selected_mode = 'Ref2VA' if meta['references'] else 'T2VA'
            if record['command'][4] != '-':
                assert f'mode={selected_mode}, explicit references={meta["references"]},' in log
            saved = state(out/(case+'.h3av')); states[case] = saved
            assert saved['frames'] == meta['raw_frames'] == 90
            assert meta['latent_callbacks'] == meta['steps'] + 1
            frames = meta['frames']; width, height = meta['width'], meta['height']
            assert frames == (90 if case == 'trim-debug' else 90 if case == 'source' else 51)
            assert meta['audio_samples'] == frames * 32000 // 24
            probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(out/(case+'.mp4'))], text=True))
            streams = probe['streams']; assert {s['codec_type'] for s in streams} == {'audio','video'}
            assert all(float(s['start_time']) == 0 and abs(float(s['duration'])-frames/24) < .001 for s in streams)
            pixels[case] = np.fromfile(out/(case+'.rgb'),dtype=np.uint8).reshape(frames,height,width,3)
            audio[case] = np.fromfile(out/(case+'.pcm'),dtype='<f4').reshape(2,meta['audio_samples'])
            assert np.isfinite(audio[case]).all()
            if meta['continuation_mode'] == 'bridge': bridge_cases.append(case)
            result['renders'][case] = {'frames':frames,'av_seconds':frames/24,'state_frames':saved['frames'],
                                      'steps':meta['steps'],'references':meta['references'],'mode':meta['continuation_mode']}
            indices = [0, frames//2, frames-1] if case != 'trim-debug' else [0,25,38,39,64,89]
            images = []
            for i in indices:
                image = case+f'-{i:03d}.png'; Image.fromarray(pixels[case][i]).save(review/image)
                images.append(f'<figure><img src="{image}"><figcaption>Frame {i}</figcaption></figure>')
            cards.append(f'<section><h2>{html.escape(case)}</h2><p>{meta["continuation_mode"]}, {meta["references"]} references, '
                         f'{meta["steps"]} steps, {frames/24:.3f} seconds</p><p>{html.escape(record["command"][10])}</p>'
                         f'<video controls preload="none" src="../{case}.mp4"></video><div class="frames">'+''.join(images)+'</div></section>')
        if bridge_cases:
            assert 'uses CPU F32 Euler sampler' in log
            assert 'bridge modulation preparation' in log and 'bridge AV initialization' in log
            assert 'audio bridge ticks=' in log and 'packed rows' in log
            result['diagnostics'][name] = diagnostics(log, meta['steps'], len(bridge_cases))
            result['profiles'][name] = profile_metrics(log, len(bridge_cases), states[bridge_cases[0]])
    if {'trim-normal','trim-debug'} <= states.keys():
        assert states['trim-normal']['data'] == states['trim-debug']['data']
        assert np.array_equal(pixels['trim-normal'],pixels['trim-debug'][39:])
        assert np.array_equal(audio['trim-normal'],audio['trim-debug'][:,52000:])
        result['trim_parity'] = 'exact full state, RGB suffix and planar stereo PCM suffix'
    if {'zero-hard','zero-bridge'} <= states.keys():
        assert states['zero-hard']['data'] == states['zero-bridge']['data']
        assert np.array_equal(pixels['zero-hard'],pixels['zero-bridge'])
        assert np.array_equal(audio['zero-hard'],audio['zero-bridge'])
        result['zero_strength_parity'] = 'exact full state, RGB and PCM'
    if {'bridge-pose','bridge-unchanged'} <= states.keys():
        pose, unchanged = states['bridge-pose'],states['bridge-unchanged']
        assert records['bridge-pose']['source_sha256'] == records['bridge-unchanged']['source_sha256']
        assert records['bridge-pose']['command'][3:10] == records['bridge-unchanged']['command'][3:10]
        assert np.array_equal(pose['video'][:,8:12],unchanged['video'][:,8:12])
        assert np.array_equal(pose['audio'][:,:,43:65],unchanged['audio'][:,:,43:65])
        v_delta=float(np.sqrt(np.mean((pose['video'][:,12:]-unchanged['video'][:,12:])**2)))
        a_delta=float(np.sqrt(np.mean((pose['audio'][:,:,65:]-unchanged['audio'][:,:,65:])**2)))
        assert v_delta > 0 and a_delta > 0
        result['changed_prompt']={'same_references_source_seed':True,'exact_rows_equal':True,
                                  'generated_video_latent_rmse':v_delta,'generated_audio_latent_rmse':a_delta}
    if args.require_complete:
        assert len(states) >= 14 and all(r['steps']==20 for r in result['renders'].values())
        assert result.get('trim_parity') and result.get('zero_strength_parity') and result.get('changed_prompt')
        for name in ['bridge-to-bridge','bridge-to-hard']:
            assert records[name]['source_sha256'] == hashlib.sha256(states['bridge-pose']['data']).hexdigest()
        assert result['renders']['t2va']['references'] == 0
    result['passed'] = True
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    (review/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Bridge integration review</title>'
        '<style>body{font:16px system-ui;background:#151719;color:#eee;margin:2rem}section{border-top:1px solid #555;padding:1rem 0}video{width:256px}.frames{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0}img{width:192px}figcaption{font-size:12px}</style>'
        '<h1>Bridge integration — T021–T042</h1><p>Numerical integration checks; motion-quality acceptance and parameter sweeps remain later tasks.</p>'+''.join(cards))
    print(f'ok: {len(states)} renders; exact-row, velocity scaling, state, AV timing and available paired checks passed')


if __name__ == '__main__': main()
