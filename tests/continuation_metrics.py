#!/usr/bin/env python3
"""Measure raw acceptance RGB/PCM, validate trim/state invariants, build review HTML.

Only Python's standard library and FFmpeg/ffprobe are needed. Color statistics
describe scene evolution; they do not alone establish identity or motion quality.
"""
import argparse
from array import array
import hashlib
import html
import json
import math
from pathlib import Path
import struct
import subprocess

def state(path):
    blob=path.read_bytes()
    assert blob[:8]==b'H3AV\r\n\x1a\n'
    assert hashlib.sha256(blob[:128]+blob[160:]).digest()==blob[128:160]
    geometry=struct.unpack_from('<10I',blob,24)
    nv,na=struct.unpack_from('<QQ',blob,72)
    assert len(blob)==160+nv+na
    # Acceptance renders must be numerically valid; serialization itself also
    # supports lossless special F32 bit patterns tested by the host suite.
    values=array('f');values.frombytes(blob[160:])
    assert all(math.isfinite(v) for v in values),'non-finite rendered AV latent'
    return geometry,blob[160:160+nv],blob[160+nv:]

def prefix_bytes(s,context=39):
    g,v,a=s; vt,h,w,at=g[3:7]; k=(context-39)//51
    vp,ap=12+15*k,65+85*k
    return (b''.join(v[c*vt*h*w*4:(c*vt+vp)*h*w*4] for c in range(24)),
            b''.join(a[c*at*4:(c*at+ap)*4] for c in range(64)))

def frame_stats(rgb):
    n=len(rgb)//3
    means=[sum(rgb[c::3])/n for c in range(3)]
    hist=[0]*32; saturation=0; luma_sum=0; luma_sq=0; samples=0
    skin_sum=[0,0,0]; skin_count=0
    # Fixed spatial sampling preserves comparability while keeping this script
    # cheap without NumPy. Means use every pixel.
    for i in range(0,len(rgb),3*4):
        r,g,b=rgb[i:i+3]; maximum=max(r,g,b)
        saturation+=(maximum-min(r,g,b))/maximum if maximum else 0
        y=0.2126*r+0.7152*g+0.0722*b
        luma_sum+=y; luma_sq+=y*y; hist[min(31,int(y/8))]+=1; samples+=1
        # A fixed, deliberately simple skin-color proxy separates skin tone
        # from changing amounts of green scenery. This is not segmentation or
        # an identity score; inspect its coverage and the rendered frames.
        if r>95 and g>40 and b>20 and r>g+10 and r>b and maximum-min(r,g,b)>15:
            skin_count+=1
            for c,value in enumerate([r,g,b]): skin_sum[c]+=value
    r,g,b=means; y=0.2126*r+0.7152*g+0.0722*b
    return {'rgb':means,'luma':y,'chroma_cb':(b-y)/1.8556,'chroma_cr':(r-y)/1.5748,
            'saturation':saturation/samples,'luma_std':math.sqrt(max(0,luma_sq/samples-(luma_sum/samples)**2)),
            'histogram':[v/samples for v in hist],
            'skin_proxy_rgb':[v/skin_count for v in skin_sum] if skin_count else None,
            'skin_proxy_coverage':skin_count/samples}

def rmse(a,b):
    assert len(a)==len(b) and len(a)>0
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a))

def png(rgb,width,height,path):
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
                    '-video_size',f'{width}x{height}','-i','pipe:0','-frames:v','1',str(path)],input=rgb,check=True)

def plot_chains(chains,review):
    if len(chains)!=2: return False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fields=[('rgb_drift','Mean RGB drift (0–255)'),('luma_drift','Luma change (0–255)'),
            ('chroma_cb_drift','Cb change'),('chroma_cr_drift','Cr change'),
            ('saturation','HSV saturation'),('boundary_histogram_distance','Boundary histogram distance')]
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,(field,label) in zip(axes.flat,fields):
        for method,color in [('native','#1976b9'),('recursive','#e07827')]:
            rows=chains[method]
            ax.plot([r['segment'] for r in rows],[r[field] for r in rows],marker='o',label=method,color=color)
        ax.set(xlabel='Generation segment',ylabel=label,xticks=range(1,6));ax.grid(alpha=.25)
    axes[0,0].legend();fig.suptitle('Five-segment H3 continuation: raw RGB measurements')
    fig.savefig(review/'color-drift.png',dpi=180);plt.close(fig)
    return True

def contact_sheet(names,records,frames,review,label):
    """Render a labeled temporal comparison using the measured raw frames."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    names=[n for n in names if n in records]
    if not names: return False
    fig,axes=plt.subplots(len(names),3,figsize=(9,3*len(names)),squeeze=False,layout='constrained')
    for row,name in enumerate(names):
        r=records[name]; raw,size=frames[name]
        for col,index in enumerate([0,r['frames']//2,r['frames']-1]):
            rgb=np.frombuffer(raw[index*size:(index+1)*size],dtype=np.uint8).reshape(r['height'],r['width'],3)
            axes[row,col].imshow(rgb);axes[row,col].axis('off')
            axes[row,col].set_title(f'{name} / frame {index}',fontsize=10)
    fig.savefig(review/(label+'.png'),dpi=120);plt.close(fig)
    return True

def plot_audio_seam(source_prefix,debug_prefix,source,normal,review):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    a=np.asarray(source_prefix).reshape(2,-1);b=np.asarray(debug_prefix).reshape(2,-1)
    prior=np.asarray(source).reshape(2,-1);new=np.asarray(normal).reshape(2,-1)
    fig,axes=plt.subplots(2,1,figsize=(10,6),layout='constrained')
    for channel,label in enumerate(['Left','Right']):
        starts=range(0,a.shape[1],256)
        errors=[max(1e-10,float(np.sqrt(np.mean((a[channel,i:i+256]-b[channel,i:i+256])**2)))) for i in starts]
        axes[0].plot(np.asarray(list(starts))/32000,errors,label=label)
        joined=np.concatenate([prior[channel,-160:],new[channel,:160]])
        axes[1].plot(np.arange(-160,160)/32,joined,label=label)
    axes[0].set(xlabel='Time in protected prefix (s)',ylabel='Window RMS error',yscale='log',title='Source tail versus debug prefix (256-sample windows)')
    axes[1].axvline(0,color='black',linewidth=.8)
    axes[1].set(xlabel='Time from delivered audio join (ms)',ylabel='F32 PCM',title='Previous segment end and new delivered audio start')
    for ax in axes: ax.grid(alpha=.25);ax.legend()
    fig.savefig(review/'audio-seam.png',dpi=150);plt.close(fig)
    return True

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,nargs='?',default=Path('outputs/continuation-validation/acceptance'))
    parser.add_argument('--require-complete',action='store_true',help='Require all 20-step matrix cases, paired debug output and CLI resume')
    args=parser.parse_args(); root=args.directory.resolve(); review=root/'review'; review.mkdir(exist_ok=True)
    records={}; frames={}; report={'segments':records}
    for meta_path in sorted(root.glob('*.json')):
        meta=json.loads(meta_path.read_text()); name=meta_path.stem
        if 'raw_frames' not in meta: continue
        rgb=(root/(name+'.rgb')).read_bytes(); size=meta['width']*meta['height']*3
        assert len(rgb)==size*meta['frames']
        assert (root/(name+'.pcm')).stat().st_size==meta['audio_samples']*2*4
        g,_,_=state(root/(name+'.h3av')); assert g[2]==meta['raw_frames']
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(root/(name+'.mp4'))]))
        streams={s['codec_type']:s for s in probe['streams']}
        assert abs(float(streams['video']['start_time']))<1e-6
        assert abs(float(streams['audio']['start_time']))<1/32000
        assert int(streams['video']['nb_frames'])==meta['frames']
        assert abs(float(streams['video']['duration'])-meta['frames']/24)<1e-5
        assert abs(float(streams['audio']['duration'])-meta['audio_samples']/32000)<1/32000+1e-5
        indices=[0,meta['frames']//2,meta['frames']-1]
        stats=[frame_stats(rgb[i*size:(i+1)*size]) for i in range(meta['frames'])]
        record=dict(meta)
        for key in ['luma','chroma_cb','chroma_cr','saturation','luma_std']:
            record[key]=sum(f[key] for f in stats)/len(stats)
        record['rgb']=[sum(f['rgb'][c] for f in stats)/len(stats) for c in range(3)]
        skin=[f['skin_proxy_rgb'] for f in stats if f['skin_proxy_rgb'] is not None]
        record['skin_proxy_rgb']=[sum(f[c] for f in skin)/len(skin) for c in range(3)] if skin else None
        record['skin_proxy_coverage']=sum(f['skin_proxy_coverage'] for f in stats)/len(stats)
        record['first']=stats[0]; record['last']=stats[-1]; record['per_frame']=stats
        records[name]=record; frames[name]=(rgb,size)
        for label,index in zip(['first','middle','last'],indices):
            png(rgb[index*size:(index+1)*size],meta['width'],meta['height'],review/(name+'-'+label+'.png'))
    if args.require_complete:
        from run_continuation import cases
        expected={name for name,_ in cases(root,'',20)}
        assert expected|{'debug-prefix'}<=records.keys(),'acceptance renders are missing'
        runs=json.loads((root/'runs.json').read_text())
        assert all(runs.get(name,{}).get('returncode')==0 for name in expected),'matrix case failed or unrecorded'
        assert all(records[name]['steps']==20 and records[name]['latent_callbacks']==21
                   for name in expected|{'debug-prefix'}),'acceptance requires all 20 transitions'
        resumed=state(root/'cli/resume.h3av')
        assert prefix_bytes(resumed)==prefix_bytes(state(root/'native-2.h3av'))
        assert 'continuation uses CPU F32 Euler sampler' in (root/'cli/resume.log').read_text()
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(root/'cli/resume.mp4')]))
        streams={s['codec_type']:s for s in probe['streams']}
        assert int(streams['video']['nb_frames'])==51
        for kind in ['video','audio']:
            assert abs(float(streams[kind]['start_time']))<1e-6
            assert abs(float(streams[kind]['duration'])-2.125)<1/32000
        report['complete_matrix']={'cases':len(expected),'debug_pair':True,'cli_resume':True,'steps':20}
    if all(name in records for name in ['native-1','trim-audit','debug-prefix']):
        a=state(root/'trim-audit.h3av'); b=state(root/'debug-prefix.h3av')
        assert a==b,'keep-prefix must not affect complete state'
        normal,size=frames['trim-audit']; debug,_=frames['debug-prefix']; source,_=frames['native-1']
        assert normal==debug[39*size:],'video trimming must be exact'
        tail=source[-39*size:]; prefix=debug[:39*size]
        audio_source=array('f'); audio_source.frombytes((root/'native-1.pcm').read_bytes())
        audio_debug=array('f'); audio_debug.frombytes((root/'debug-prefix.pcm').read_bytes())
        audio_normal=array('f'); audio_normal.frombytes((root/'trim-audit.pcm').read_bytes())
        n_source=len(audio_source)//2; n_debug=len(audio_debug)//2
        a_tail=array('f'); a_prefix=array('f'); a_suffix=array('f')
        for c in range(2):
            a_tail.extend(audio_source[(c+1)*n_source-52000:(c+1)*n_source])
            a_prefix.extend(audio_debug[c*n_debug:c*n_debug+52000])
            a_suffix.extend(audio_debug[c*n_debug+52000:(c+1)*n_debug])
        assert a_suffix.tobytes()==audio_normal.tobytes(),'audio trimming must be exact'
        report['seam']={'video_prefix_rmse_255':rmse(tail,prefix),'audio_prefix_rmse':rmse(a_tail,a_prefix),
                        'context_frames':39,'context_ticks':65,'context_samples':52000,
                        'state_identical_with_keep':True,'rgb_trim_exact':True,'pcm_trim_exact':True}
        report['seam']['video_first_five_frames_rmse_255']=rmse(tail[:5*size],prefix[:5*size])
        report['seam']['video_remaining_context_rmse_255']=rmse(tail[5*size:],prefix[5*size:])
        report['seam']['video_context_per_frame_rmse_255']=[
            rmse(tail[i*size:(i+1)*size],prefix[i*size:(i+1)*size]) for i in range(39)]
        report['seam']['audio_source_prefix_rms']=rmse(a_tail,[0.0]*len(a_tail))
        # Receptive-field edges can differ despite identical audio latents.
        # Record interior error and the actual join without hiding edge error.
        interior_source=array('f'); interior_debug=array('f')
        for c in range(2):
            interior_source.extend(a_tail[c*52000+8000:(c+1)*52000-8000])
            interior_debug.extend(a_prefix[c*52000+8000:(c+1)*52000-8000])
        report['seam']['audio_interior_rmse']=rmse(interior_source,interior_debug)
        report['seam']['audio_join_step']=[audio_normal[c*(len(audio_normal)//2)]-audio_source[(c+1)*n_source-1] for c in range(2)]
        report['seam']['audio_plot']=plot_audio_seam(a_tail,a_prefix,audio_source,audio_normal,review)
        if args.require_complete:
            # Fixture tolerances allow VAE receptive-field edges while catching
            # timeline shifts or a decoded/re-encoded continuation source.
            seam=report['seam'];rms=seam['audio_source_prefix_rms']
            assert seam['video_prefix_rmse_255']<5
            assert seam['audio_prefix_rmse']<max(1e-6,0.25*rms)
            assert seam['audio_interior_rmse']<max(1e-6,0.02*rms)
        (review/'source-tail.rgb').write_bytes(tail); (review/'debug-prefix.rgb').write_bytes(prefix)
        for label,raw in [('source-tail',tail),('copied-prefix',prefix)]:
            png(raw[:size],records['native-1']['width'],records['native-1']['height'],review/(label+'.png'))
        for name in ['changed-reference','changed-prompt','reuse-2','reuse-3']:
            if name in records:
                changed=state(root/(name+'.h3av'))
                assert prefix_bytes(a)==prefix_bytes(changed),name+' altered protected latents'
                if name in ['changed-reference','changed-prompt']:
                    assert a[1:]!=changed[1:],name+' did not change generated suffix'
    chains={}
    for method in ['native','recursive']:
        names=['native-1']+[f'{method}-{n}' for n in range(2,6)]
        if not all(n in records for n in names): continue
        reference=records[names[0]]
        rows=[]
        for i,name in enumerate(names):
            r=records[name]; rgb_drift=rmse(r['rgb'],reference['rgb'])
            boundary=None
            if i:
                prior=records[names[i-1]]['last']['histogram']; current=r['first']['histogram']
                boundary=sum(abs(a-b) for a,b in zip(prior,current))/2
            rows.append({'segment':i+1,'name':name,'rgb_drift':rgb_drift,
                         'luma_drift':r['luma']-reference['luma'],'chroma_cb_drift':r['chroma_cb']-reference['chroma_cb'],
                         'chroma_cr_drift':r['chroma_cr']-reference['chroma_cr'],
                         'contrast_drift':r['luma_std']-reference['luma_std'],
                         'skin_proxy_rgb_drift':rmse(r['skin_proxy_rgb'],reference['skin_proxy_rgb']) if r['skin_proxy_rgb'] and reference['skin_proxy_rgb'] else None,
                         'saturation':r['saturation'],'boundary_histogram_distance':boundary})
        chains[method]=rows
    report['chains']=chains
    plotted=plot_chains(chains,review)
    sheets=[]
    for label,names in [('native-chain',['native-1']+[f'native-{n}' for n in range(2,6)]),
                        ('recursive-chain',['native-1']+[f'recursive-{n}' for n in range(2,6)]),
                        ('changed-conditioning',['trim-audit','changed-reference','changed-prompt']),
                        ('reuse-comparison',['trim-audit','reuse-2','reuse-3'])]:
        if contact_sheet(names,records,frames,review,label): sheets.append(label)
    if len(chains)==2:
        report['drift_comparison']={m:sum(r['rgb_drift'] for r in rows[1:])/4 for m,rows in chains.items()}
    (root/'metrics.json').write_text(json.dumps(report,indent=2))
    cards=[]
    for name,r in records.items():
        figures=''.join(f'<figure><img src="{name}-{label}.png"><figcaption>{label}</figcaption></figure>' for label in ['first','middle','last'])
        cards.append(f'<article><h2>{html.escape(name)}</h2><p>{r["frames"]} delivered frames; {r["audio_samples"]} PCM samples; '
                     f'luma {r["luma"]:.1f}; saturation {r["saturation"]:.3f}</p><div>{figures}</div>'
                     f'<video controls preload="none" src="../{name}.mp4"></video></article>')
    document='''<!doctype html><meta charset="utf-8"><title>H3 continuation acceptance</title>
<style>body{font:16px system-ui;background:#17191d;color:#eee;margin:2rem;max-width:1100px}article{border-top:1px solid #555;padding:1rem 0}article div{display:flex;gap:1rem}figure{margin:0}img{width:256px;max-width:100%}video{width:512px;max-width:100%;margin-top:1rem}p{line-height:1.5}</style>
<h1>H3 continuation acceptance</h1><p>Review identity, skin tone, background color, camera motion, body motion, and exposure at each boundary.
Compare native-1 → native-5 with native-1 → recursive-2 → recursive-5. Inspect source-tail and copied-prefix separately.
Numerical color changes can reflect legitimate scene motion. Record visual decisions in the acceptance report.</p>'''
    if plotted: document+='<img style="width:100%" src="color-drift.png" alt="Native versus recursive color drift measurements">'
    document+='<p>'+ ' · '.join(f'<a style="color:#8bd" href="{label}.png">{label}</a>' for label in sheets)+'</p>'
    if 'seam' in report:
        document+='<article><h2>Copied history</h2><div><figure><img src="source-tail.png"><figcaption>Source tail, first frame</figcaption></figure><figure><img src="copied-prefix.png"><figcaption>Debug prefix, first frame</figcaption></figure></div>'
        document+='<p>The 39-frame video and 52,000-sample stereo prefixes align exactly in duration. VAE context edges account for small reconstruction differences.</p>'
        if report['seam']['audio_plot']: document+='<img style="width:100%" src="audio-seam.png" alt="Audio context error and delivered boundary waveform">'
        document+='</article>'
    document+=''.join(cards)
    (review/'index.html').write_text(document)
    print(json.dumps({k:v for k,v in report.items() if k!='segments'},indent=2))
    print(f'ok: measured {len(records)} renders; review {review/"index.html"}')

if __name__=='__main__': main()
