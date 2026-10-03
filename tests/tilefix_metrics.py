#!/usr/bin/env python3
"""Pixel/11x11 luminance SSIM and diagnostic overlap-boundary seam measurements.

SSIM uses valid uniform windows, population covariance, L=1, K1=.01,K2=.03.
Seam ratios are descriptive, not correctness thresholds: content edges count too.
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
LUMA=np.array([.2126,.7152,.0722],dtype=np.float64)
def box(a,n=11):
    s=np.pad(a,((1,0),(1,0))).cumsum(0).cumsum(1)
    return (s[n:,n:]-s[:-n,n:]-s[n:,:-n]+s[:-n,:-n])/(n*n)
def compare(a,b):
    if a.shape!=b.shape:raise ValueError('comparison shapes differ')
    absolute=0.;square=0.;maximum=0.;ssim=0.;count=0
    for av,bv in zip(a,b):
        av=av.astype(np.float64);bv=bv.astype(np.float64);d=av-bv
        absolute+=np.abs(d).sum();square+=(d*d).sum();maximum=max(maximum,float(np.abs(d).max()));count+=d.size
        x=av@LUMA;y=bv@LUMA;mx=box(x);my=box(y)
        vx=np.maximum(box(x*x)-mx*mx,0);vy=np.maximum(box(y*y)-my*my,0);cov=box(x*y)-mx*my
        ssim+=float((((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).mean())
    mse=square/count
    return {'mae':absolute/count,'max_abs':maximum,'rmse':float(np.sqrt(mse)),
            'psnr_db':float(-10*np.log10(mse)) if mse else None,'exact':bool(mse==0),'ssim_luma_11x11':ssim/len(a)}
def seams(rgb,plan):
    result={}
    for axis,key in ((1,'y'),(2,'x')):
        p=plan[key];extent=rgb.shape[axis]
        boundaries=sorted(set(p['starts'][1:]+[s+p['length'] for s in p['starts'][:-1]]))
        lum=np.asarray(rgb,dtype=np.float32)@LUMA.astype(np.float32)
        profile=np.abs(np.diff(lum,axis=axis)).mean(axis=tuple(i for i in range(3) if i!=axis))
        rows=[]
        for pos in boundaries:
            nearby=[j for j in range(max(1,pos-12),min(extent,pos+13)) if abs(j-pos)>=3 and j not in boundaries]
            edge=float(profile[pos-1]);base=float(profile[np.array(nearby)-1].mean()) if nearby else 0
            def step(k):
                left=np.take(lum,range(max(0,k-4),k),axis=axis).mean(axis=axis)
                right=np.take(lum,range(k,min(extent,k+4)),axis=axis).mean(axis=axis)
                return float(np.abs(right-left).mean())
            jump=step(pos);jumpbase=float(np.mean([step(k) for k in nearby])) if nearby else 0
            rows.append({'position':pos,'gradient':edge,'nearby_gradient':base,'gradient_ratio':edge/max(base,1e-12),
                'luminance_step':jump,'nearby_luminance_step':jumpbase,'luminance_ratio':jump/max(jumpbase,1e-12)})
        result[key]={'boundaries':rows,'mean_gradient_ratio':float(np.mean([r['gradient_ratio'] for r in rows])) if rows else None,
                     'mean_luminance_ratio':float(np.mean([r['luminance_ratio'] for r in rows])) if rows else None}
    return result

def media(d,arrays,h,w):
    labels=list(arrays);thumbw=min(w,384);thumbh=round(h*thumbw/w)
    sheet=Image.new('RGB',(thumbw*len(labels),(thumbh+24)*3),'white');draw=ImageDraw.Draw(sheet)
    frames=[0,len(next(iter(arrays.values())))//2,len(next(iter(arrays.values())))-1]
    for col,label in enumerate(labels):
        a=arrays[label]
        for row,t in enumerate(frames):
            image=Image.fromarray(np.rint(np.clip(a[t],0,1)*255).astype(np.uint8))
            if row==1:image.save(d/(label+'-middle.png'))
            sheet.paste(image.resize((thumbw,thumbh)),(col*thumbw,row*(thumbh+24)+24))
            draw.text((col*thumbw+3,row*(thumbh+24)+3),f'{label}, frame {t}',fill='black')
        movie=d/(label+'.mp4')
        if not movie.exists():
            proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','24','-i','-','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p',str(movie)],stdin=subprocess.PIPE)
            for frame in a:proc.stdin.write(np.rint(np.clip(frame,0,1)*255).astype(np.uint8).tobytes())
            proc.stdin.close();assert proc.wait()==0
    sheet.save(d/'comparison.png')
    # Temporal mean absolute error reveals tile-aligned persistent structure.
    if 'official' in arrays:
        for label in ('default','auto','320'):
            diff=np.abs(arrays[label]-arrays['official']).mean(axis=0)
            Image.fromarray(np.rint(np.clip(diff*8,0,1)*255).astype(np.uint8)).save(d/(label+'-error-x8.png'))
        # Frame-by-frame flat-band luminance: moving input lines are diagonal;
        # stationary vertical reconstruction bias remains in fixed columns.
        timeline=Image.new('RGB',(w,24+len(arrays)*200),'white');draw=ImageDraw.Draw(timeline)
        draw.text((4,4),'Flat-band temporal slices: frames 0..21 downward; luminance error x16 centered on gray',fill='black')
        for i,(label,a) in enumerate(arrays.items()):
            draw.text((4,26+i*200),label,fill='black')
            if label=='official': profile=(a[:,h*3//4+24:h-8]@LUMA).mean(axis=1)
            else: profile=.5+16*((a[:,h*3//4+24:h-8]-arrays['official'][:,h*3//4+24:h-8])@LUMA).mean(axis=1)
            strip=Image.fromarray(np.rint(np.clip(profile,0,1)*255).astype(np.uint8)).convert('RGB').resize((w,176),Image.Resampling.NEAREST)
            timeline.paste(strip,(0,48+i*200))
        timeline.save(d/'motion-slices.png')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    summary={}
    for d in sorted(a.directory.iterdir()):
        if not (d/'runs.json').exists():continue
        h,w=map(int,d.name.split('x'));records=json.loads((d/'runs.json').read_text())
        if not all(mode in records for mode in ('default','auto','320')):continue
        arrays={mode:np.memmap(d/(mode+'.f32'),dtype='<f4',mode='r').reshape(-1,h,w,3) for mode in ('official','default','auto','320') if mode in records and (d/(mode+'.f32')).exists()}
        results={'seams':{mode:seams(arrays[mode],records[mode]) for mode in ('default','auto','320')},'comparisons':{}}
        for left,right in [('default','official'),('auto','official'),('320','official'),('default','auto'),('default','320'),('auto','320')]:
            if left in arrays and right in arrays:results['comparisons'][left+'-vs-'+right]=compare(arrays[left],arrays[right])
        (d/'metrics.json').write_text(json.dumps(results,indent=2)+'\n');summary[d.name]=results
        media(d,arrays,h,w);print(d.name,results['comparisons'].get('default-vs-official'),flush=True)
    (a.directory/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
