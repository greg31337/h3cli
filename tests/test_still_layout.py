#!/usr/bin/env python3
"""Independent direct-still coordinates and unchanged h3cli RNG/sigma semantics."""
import json,math,subprocess,struct
from pathlib import Path

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def grid(h,w,t):
 ratio_h=h/math.sqrt(h*w);ratio_w=w/math.sqrt(h*w)
 return [[t,((y/(h//2))*ratio_h+(1-ratio_h)/2)*32,((x/(w//2))*ratio_w+(1-ratio_w)/2)*32] for y in range(h//2) for x in range(w//2)]
def expected(refs):
 rows=[[i,0,0] for i in range(11)];segments=[['text',0,11]];cursor=11
 for h,w in [(16,16),(20,32)][:refs]:
  start=len(rows);rows+=grid(h,w,cursor);segments.append(['ref_img',start,len(rows)]);cursor+=1
 target=grid(30,40,cursor);low=target[0][2];high=target[19][2];start=len(rows)
 rows += [[cursor+i,0,x] for x in [low,high] for i in range(2)];segments.append(['audio',start,len(rows)]);start=len(rows)
 rows += target;segments.append(['video',start,len(rows)]);return rows,segments

def rng(seed):
 state=0;increment=2*seed+1;mask=2**64-1
 def u32():
  nonlocal state
  old=state;state=(old*6364136223846793005+increment)&mask;x=(((old>>18)^old)>>27)&0xffffffff;r=old>>59
  return ((x>>r)|(x<<((-r)&31)))&0xffffffff
 u32();state=(state+(seed^0x9e3779b97f4a7c15))&mask;u32()
 while True:
  a=(u32()+1)/4294967297;b=(u32()+.5)/4294967296;r=math.sqrt(-2*math.log(a));angle=2*math.pi*b
  yield f32(r*math.cos(angle));yield f32(r*math.sin(angle))

def main():
 root=Path('outputs/single-still/layout');root.mkdir(parents=True,exist_ok=True)
 for refs in range(3):
  got=json.loads(subprocess.check_output(['./bin/still_layout_fixture',str(refs)]));rows,segments=expected(refs)
  assert got['segments']==segments;assert len(rows)==len(got['positions'])
  assert max(abs(a-b) for x,y in zip(rows,got['positions']) for a,b in zip(x,y))<1e-12
  for modality,shift in [('video',12),('audio',3)]:
   sig=[]
   for i in range(6):
    base=f32((1000-i*1000//6)/1000);sig.append(f32(f32(shift*base)/f32(1+f32((shift-1)*base))))
   sig.append(0);assert max(abs(a-b) for a,b in zip(sig,got[modality+'_sigmas']))<1e-8
  r=rng(42);assert max(abs(next(r)-x) for x in got['noise'])<1e-8
  (root/f'references-{refs}.json').write_text(json.dumps(got,indent=2)+'\n')
 print('PASS independent T=1 positions, ordered image slots, four audio rows, sigmas and seed-42 noise')
if __name__=='__main__':main()
