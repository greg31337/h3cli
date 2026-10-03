#!/usr/bin/env python3
"""Measure and extract the last source second and first two new seconds.

Optical flow is a motion proxy, not an identity or action classifier. Selection
gates are declared in docs/bridge-quality-protocol.json and visual review is
required before claiming quality acceptance.
"""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

from bridge_metrics import state

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/bridge-quality-protocol.json'
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sidecar(base, extension):
    return Path(str(base)+'.'+extension)


def frames(base):
    meta=json.loads(sidecar(base,'json').read_text())
    data=np.fromfile(sidecar(base,'rgb'),np.uint8).reshape(meta['frames'],meta['height'],meta['width'],3)
    return data,meta


def upper_mean(values):
    values=np.asarray(values).ravel()
    count=max(1,len(values)//10)
    return float(np.mean(np.partition(values,len(values)-count)[-count:]))


def regions(height,width,protocol):
    y,x=np.mgrid[:height,:width]
    x0,y0,x1,y1=protocol['subject_roi_xyxy']
    subject=(x>=x0*width)&(x<x1*width)&(y>=y0*height)&(y<y1*height)
    background=((y<.23*height)|(x<.25*width))&(x>=4)&(x<width-4)&(y>=4)&(y<height-4)
    return subject,background


def camera_flow(before,after,background,flow):
    """Track background features; report tracking coverage and affine residual."""
    points=cv2.goodFeaturesToTrack(before,100,.01,5,mask=background.astype(np.uint8)*255)
    affine=None; tracked=0
    if points is not None:
        moved,status,_=cv2.calcOpticalFlowPyrLK(before,after,points,None,winSize=(15,15),maxLevel=2)
        if moved is not None:
            valid=status.ravel().astype(bool)&np.isfinite(moved.reshape(-1,2)).all(axis=1)
            tracked=int(valid.sum())
            if tracked>=6:
                cv2.setRNGSeed(0)
                affine,_=cv2.estimateAffinePartial2D(points[valid],moved[valid],method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=1000)
    height,width=before.shape; y,x=np.mgrid[:height,:width].astype(np.float32)
    if affine is None:
        translation=np.median(flow[background],axis=0)
        field=np.empty_like(flow); field[:]=translation
    else:
        field=np.stack([(affine[0,0]-1)*x+affine[0,1]*y+affine[0,2],
                        affine[1,0]*x+(affine[1,1]-1)*y+affine[1,2]],axis=-1).astype(np.float32)
    return field,tracked


def motion_measurements(rgb,protocol):
    gray=np.stack([cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) for frame in rgb])
    height,width=gray.shape[1:]; subject,background=regions(height,width,protocol)
    y,x=np.mgrid[:height,:width].astype(np.float32)
    settings={k:v for k,v in protocol['flow'].items() if k!='algorithm'}
    flows=[]; residuals=[]; background_motion=[]; camera=[]; tracks=[]
    for a,b in zip(gray[:-1],gray[1:]):
        flow=cv2.calcOpticalFlowFarneback(a,b,None,**settings)
        field,tracked=camera_flow(a,b,background,flow)
        residual=flow-field
        flows.append(flow); residuals.append(residual)
        background_motion.append(upper_mean(np.linalg.norm(residual[background],axis=1)))
        camera.append(float(np.sqrt(np.mean(field[background]**2))))
        tracks.append(tracked)
    accelerations=[]; body_acc=[]; body_jerk=[]
    for i in range(len(flows)-1):
        carried=cv2.remap(residuals[i+1],x+flows[i][:,:,0],y+flows[i][:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        acceleration=carried-residuals[i]; accelerations.append(acceleration)
        body_acc.append(upper_mean(np.linalg.norm(acceleration[subject],axis=1)))
    for i in range(len(accelerations)-1):
        carried=cv2.remap(accelerations[i+1],x+flows[i][:,:,0],y+flows[i][:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        body_jerk.append(upper_mean(np.linalg.norm((carried-accelerations[i])[subject],axis=1)))
    body_motion=[upper_mean(np.linalg.norm(f[subject],axis=1)) for f in residuals]
    direction=[np.mean(f[subject],axis=0).tolist() for f in residuals]
    # Frame 24 is the first new frame; include the seam acceleration/jerk.
    acc=np.asarray(body_acc[22:]); jerk=np.asarray(body_jerk[21:])
    spike_threshold=max(.05,2*float(np.quantile(body_acc[:21],.95)))
    spikes=acc>spike_threshold
    colors=np.array([np.mean(cv2.cvtColor(frame,cv2.COLOR_RGB2LAB)[background],axis=0) for frame in rgb])
    contrast=np.array([np.std(frame[background]) for frame in gray])
    sharpness=[float(np.var(cv2.Laplacian(frame,cv2.CV_64F)[subject])) for frame in gray]
    return {
        'continuity_score':float(np.sqrt(np.mean(acc**2)+.25*np.mean(jerk**2))),
        'acceleration_rms':float(np.sqrt(np.mean(acc**2))), 'jerk_rms':float(np.sqrt(np.mean(jerk**2))),
        'spike_threshold':spike_threshold,'spike_frames':int(spikes.sum()),
        'spike_events':int(np.count_nonzero(spikes & ~np.r_[False,spikes[:-1]])),
        'motion_amplitude':float(np.mean(body_motion[23:])),
        'background_deformation':float(np.mean(background_motion[23:])),
        'camera_motion':float(np.mean(camera[23:])), 'background_feature_tracks_min':min(tracks[23:]),
        'background_color_delta':float(np.linalg.norm(np.mean(colors[24:],axis=0)-np.mean(colors[:24],axis=0))),
        'background_contrast':float(np.mean(contrast[24:])), 'sharpness':float(np.mean(sharpness[24:])),
        'luminance':[float(np.mean(frame)) for frame in gray],
        'curves':{'motion':body_motion,'direction_xy':direction,'acceleration':body_acc,'jerk':body_jerk,
                  'background_deformation':background_motion,'camera_motion':camera,
                  'background_lab':colors.tolist(),'background_contrast':contrast.tolist(),'sharpness':sharpness},
    }


def compare(hard,bridge,protocol,unchanged=False):
    g=protocol['gates']; ratio=lambda key: bridge[key]/max(hard[key],1e-12)
    checks={
        'motion_retained':ratio('motion_amplitude')>=g['motion_amplitude_ratio_min'],
        'sharpness_retained':ratio('sharpness')>=g['sharpness_ratio_min'],
        'background_stable':bridge['background_deformation']<=hard['background_deformation']*g['background_deformation_ratio_max']+g['background_deformation_absolute_slack_px'],
        'camera_stable':bridge['camera_motion']<=hard['camera_motion']*g['camera_motion_ratio_max']+g['camera_motion_absolute_slack_px'],
        'color_stable':bridge['background_color_delta']<=hard['background_color_delta']+g['background_color_delta_extra_max'],
        'contrast_stable':g['background_contrast_ratio_min']<=ratio('background_contrast')<=g['background_contrast_ratio_max'],
    }
    if unchanged:
        checks['no_continuity_degradation']=bridge['continuity_score']<=hard['continuity_score']*g['unchanged_continuity_ratio_max']+g['unchanged_continuity_absolute_slack_px']
    else: checks['material_continuity_improvement']=ratio('continuity_score')<=g['primary_continuity_ratio_max']
    return {'checks':checks,'numerical_pass':all(checks.values()),'stability_pass':all(v for k,v in checks.items() if 'continuity' not in k),
            'continuity_ratio':ratio('continuity_score'),'motion_ratio':ratio('motion_amplitude'),
            'sharpness_ratio':ratio('sharpness'),'spike_frames_hard':hard['spike_frames'],'spike_frames_bridge':bridge['spike_frames']}


def extract(rgb,folder):
    folder.mkdir(parents=True,exist_ok=True)
    for i,frame in enumerate(rgb): Image.fromarray(frame).save(folder/f'{i-24:+04d}.png')
    height,width=rgb.shape[1:3]
    indices=[0,12,23,24,30,36,42,48,54,60,66,71]
    sheet=Image.new('RGB',(width*4,(height+24)*3),'#17191d'); draw=ImageDraw.Draw(sheet)
    for slot,index in enumerate(indices):
        x=(slot%4)*width; y=(slot//4)*(height+24)
        sheet.paste(Image.fromarray(rgb[index]),(x,y+24))
        draw.text((x+5,y+5),f'{index-24:+d} frames / {(index-24)/24:+.3f}s',fill='white')
    sheet.save(folder/'contact.png')
    subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
                    '-framerate','24','-i','pipe:0','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(folder/'window.mp4')],input=rgb.tobytes(),check=True)


def quality_gate(report,selection,reviews):
    """A ranking never counts as acceptance; require held-out evidence/reviews."""
    protocol=report['protocol']; chosen=selection.get('config',{})
    keys=('length','strength','profile')
    required=[('run',protocol['selection_seed']),('run',protocol['confirmation_seed']),
              ('unchanged',protocol['selection_seed']),('unchanged',protocol['confirmation_seed'])]
    required += [(action,protocol['selection_seed']) for action in ['look-up','arms','turn-left']]
    problems=[]; selected=[]; changed=[]
    if selection.get('protocol_sha256')!=report['protocol_sha256']:
        problems.append('Selection is absent or does not identify this fixed protocol.')
    for action,seed in required:
        names=[name for name,c in report['cases'].items() if name.startswith(f'{action}-{seed}-')
               and c['config']['mode']=='bridge' and all(c['config'].get(k)==chosen.get(k) for k in keys)]
        if len(names)!=1:
            problems.append(f'Missing unique selected comparison: {action}, seed {seed}.'); continue
        name=names[0]; selected.append(name); comparison=report['comparisons'].get(name,{})
        if not comparison.get('stability_pass'): problems.append(f'{name}: stability gate failed.')
        if action in ['run','unchanged'] and not comparison.get('numerical_pass'):
            problems.append(f'{name}: primary/unchanged continuity gate failed.')
        if action!='unchanged' and seed==protocol['selection_seed']:
            changed.append(comparison.get('continuity_ratio',float('inf')))
        review=reviews.get(name,{})
        if review.get('bridge_rgb_sha256')!=report['cases'][name].get('rgb_sha256') or \
           review.get('hard_rgb_sha256')!=report['cases'].get(comparison.get('hard'),{}).get('rgb_sha256'):
            problems.append(f'{name}: missing or stale visual review.')
        if not review.get('reviewer') or not review.get('method') or not review.get('notes') or \
           any(review.get('checks',{}).get(k)!='pass' for k in protocol['visual_review_required']):
            problems.append(f'{name}: visual review did not pass every required field.')
    aggregate=math.exp(sum(math.log(max(r,1e-12)) for r in changed)/len(changed)) if len(changed)==4 else None
    if aggregate is None or aggregate>protocol['gates']['changed_action_geomean_ratio_max']:
        problems.append('Changed-action aggregate continuity gate failed or is incomplete.')
    return dict(passed=not problems,selected_comparisons=selected,changed_action_geomean_ratio=aggregate,problems=problems)


def plot_comparison(hard,bridge,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    panels=[('motion','Subject motion','px/frame',.5),('acceleration','Acceleration','px/frame²',1),
            ('jerk','Jerk','px/frame³',1.5),('camera_motion','Camera motion','px/frame',.5),
            ('background_deformation','Background deformation','px/frame',.5),('luminance','Mean luminance','0–255',0)]
    for axis,(key,title,unit,offset) in zip(axes.flat,panels):
        for case,label,color in [(hard,'Hard-39','#3168ad'),(bridge,'Bridge-39','#c25829')]:
            values=case['luminance'] if key=='luminance' else case['curves'][key]
            times=(np.arange(len(values))+offset-24)/24
            axis.plot(times,values,label=label,color=color,linewidth=1.3)
        axis.axvline(0,color='#444',linestyle='--',linewidth=.8)
        axis.set(title=title,xlabel='Seconds from first new frame',ylabel=unit,xlim=(-1,2))
        axis.grid(alpha=.2)
    axes[0,0].legend(frameon=False)
    figure.suptitle(destination.stem+' — motion proxies, not an action/identity score',fontsize=10)
    figure.savefig(destination,dpi=140); plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,nargs='?',default=ROOT/'outputs/bridge-quality/acceptance')
    parser.add_argument('--protocol',type=Path,default=PROTOCOL)
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--require-quality',action='store_true',help='Fail unless numerical and recorded visual quality gates pass')
    parser.add_argument('--select',action='store_true',help='Freeze the best stable sweep candidate before confirmation renders')
    parser.add_argument('--plots',action='store_true',help='Export comparison curves as PNGs (requires matplotlib)')
    args=parser.parse_args(); out=args.directory.resolve(); protocol=json.loads(args.protocol.read_text())
    report={'protocol':protocol,'protocol_sha256':digest(args.protocol),'opencv':cv2.__version__,'complete':True,'cases':{},'comparisons':{}}
    review=out/'review'; review.mkdir(exist_ok=True); cards=[]
    for manifest in sorted(out.glob('*-run.json')):
        run=json.loads(manifest.read_text()); report['complete'] &= run['passed']
        source=Path(run['source']); prior,source_meta=frames(source)
        assert digest(sidecar(source,'h3av'))==run['source_sha256']
        provenance=json.loads(sidecar(source,'provenance.json').read_text()) if sidecar(source,'provenance.json').exists() else {}
        source_artifacts=run.get('source_artifacts',provenance.get('outputs',{}))
        assert all(ext in source_artifacts for ext in ['h3av','rgb','json'])
        for ext,value in source_artifacts.items(): assert digest(sidecar(source,ext))==value,(source,ext)
        for name,hashes in run['outputs'].items():
            for ext,value in hashes.items(): assert digest(out/(name+'.'+ext))==value,(name,ext)
            base=out/name; current,meta=frames(base); saved=state(sidecar(base,'h3av'))
            assert saved['frames']==90 and meta['steps']==20 and meta['latent_callbacks']==21
            assert meta['frames']==51 and meta['audio_samples']==68000 and meta['references']==2
            pcm=np.fromfile(sidecar(base,'pcm'),np.float32)
            assert len(pcm)==2*meta['audio_samples'] and np.isfinite(pcm).all()
            mux=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(sidecar(base,'mp4'))]))['streams']
            assert {s['codec_type'] for s in mux}=={'video','audio'}
            assert all(float(s['start_time'])==0 and abs(float(s['duration'])-51/24)<.001 for s in mux)
            assert next(s for s in mux if s['codec_type']=='video')['nb_frames']=='51'
            assert current.shape[1:]==prior.shape[1:] and len(prior)>=24 and len(current)>=48
            rgb=np.concatenate([prior[-24:],current[:48]])
            fingerprint=hashlib.sha256(rgb.tobytes()+args.protocol.read_bytes()+Path(__file__).read_bytes()+cv2.__version__.encode()).hexdigest()
            measured=sidecar(base,'quality.json')
            data=json.loads(measured.read_text()) if measured.exists() else {}
            if data.get('fingerprint')!=fingerprint:
                data=motion_measurements(rgb,protocol); data['fingerprint']=fingerprint
                measured.write_text(json.dumps(data,indent=2)+'\n'); extract(rgb,review/name)
            report['cases'][name]=dict(data,config=run['variants'][name],seed=run['seed'],prompt=run['prompt'],source_sha256=run['source_sha256'],rgb_sha256=hashes['rgb'])
            cards.append(f'<section><h2>{html.escape(name)}</h2><p>{html.escape(run["prompt"])}</p>'
                         f'<p>Continuity score {data["continuity_score"]:.5f}; acceleration {data["acceleration_rms"]:.5f}; '
                         f'jerk {data["jerk_rms"]:.5f}; spike frames {data["spike_frames"]}; motion {data["motion_amplitude"]:.5f}</p>'
                         f'<video controls preload="none" src="{name}/window.mp4"></video><a href="{name}/contact.png"><img src="{name}/contact.png"></a></section>')
        hard=[name for name,c in run['variants'].items() if c['mode']=='hard']
        if len(hard)==1 and hard[0] in report['cases']:
            for name,c in run['variants'].items():
                if c['mode']!='bridge' or name not in report['cases']: continue
                report['comparisons'][name]=compare(report['cases'][hard[0]],report['cases'][name],protocol,name.startswith('unchanged-'))
                report['comparisons'][name]['hard']=hard[0]
    assert report['cases'], 'no completed render artifacts'
    candidates=[(name,c) for name,c in report['comparisons'].items() if name.startswith('run-72-') and c['stability_pass']]
    candidates.sort(key=lambda item:item[1]['continuity_ratio'])
    report['ranked_stable_candidates']=[{'name':name,**c} for name,c in candidates]
    if args.select:
        assert report['complete'] and candidates, 'Complete the sweep before selecting a candidate.'
        name,comparison=candidates[0]
        selected=dict(name=name,config=report['cases'][name]['config'],protocol_sha256=report['protocol_sha256'],
                      source_sha256=report['cases'][name]['source_sha256'],primary_gate_passed=comparison['numerical_pass'],
                      quality_accepted=False,selection_reason='Lowest continuity score among stable pilot candidates; confirmation and visual review still required.',
                      candidates_measured=len([n for n in report['comparisons'] if n.startswith('run-72-')]))
        target=out/'selection.json'
        if target.exists(): assert json.loads(target.read_text())==selected, 'Selection is already fixed; use a separate experiment for a new selection.'
        else: target.write_text(json.dumps(selected,indent=2)+'\n')
    selection=json.loads((out/'selection.json').read_text()) if (out/'selection.json').exists() else {}
    reviews=json.loads((out/'visual-review.json').read_text()) if (out/'visual-review.json').exists() else {}
    report['selection_sha256']=digest(out/'selection.json') if selection else None
    report['visual_review_sha256']=digest(out/'visual-review.json') if reviews else None
    report['quality_gate']=quality_gate(report,selection,reviews)
    report['quality_accepted']=report['complete'] and report['quality_gate']['passed']
    report['acceptance_note']='Visual reviews and held-out action/seed confirmation are required; candidate ranking is not acceptance.'
    if args.require_complete: assert report['complete'], 'one or more batches have not completed'
    if args.plots:
        for name,c in report['comparisons'].items():
            plot_comparison(report['cases'][c['hard']],report['cases'][name],review/(name+'-curves.png'))
            cards.append(f'<section><h2>{html.escape(name)} diagnostics</h2><img src="{name}-curves.png"></section>')
    (out/'quality.json').write_text(json.dumps(report,indent=2)+'\n')
    (review/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Bridge motion-quality review</title>'
        '<style>body{background:#17191d;color:#eee;font:16px system-ui;margin:2rem}section{border-top:1px solid #555;padding:1rem 0}img{max-width:100%;width:1024px}video{display:block;width:512px;max-width:100%}p{max-width:1000px;line-height:1.5}</style>'
        '<h1>Bridge motion-quality review</h1><p>Each window contains one source second and two delivered seconds. '
        'The join is at +0 frames. Flow metrics are proxies; review scene, character, pose, action, and camera continuity.</p>'+''.join(cards))
    print('Measured',len(report['cases']),'renders; complete:',report['complete'])
    for name,c in sorted(report['comparisons'].items(),key=lambda item:item[1]['continuity_ratio']):
        print(name,'continuity ratio',round(c['continuity_ratio'],4),'stability',c['stability_pass'],'numerical',c['numerical_pass'])
    if args.require_quality: assert report['quality_accepted'], report['quality_gate']['problems']


if __name__=='__main__': main()
