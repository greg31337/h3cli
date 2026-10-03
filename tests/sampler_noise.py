#!/usr/bin/env python3
"""Independent PCG32/Box–Muller oracle for captured original noise and RNG state."""
import array
import math
from pathlib import Path
import struct
import sys
from test_sampler_file import entries

MASK64=(1<<64)-1
class PCG:
    def __init__(self,seed):
        self.state=0; self.increment=((seed<<1)|1)&MASK64
        self.u32(); self.state=(self.state+(seed^0x9e3779b97f4a7c15))&MASK64; self.u32()
    def u32(self):
        old=self.state; self.state=(old*6364136223846793005+self.increment)&MASK64
        shifted=(((old>>18)^old)>>27)&0xffffffff; rotation=old>>59
        return ((shifted>>rotation)|(shifted<<((-rotation)&31)))&0xffffffff
    def normals(self,count):
        result=array.array('f')
        spare=0.0
        for _ in range((count+1)//2):
            u1=(self.u32()+1.0)/4294967297.0; u2=(self.u32()+0.5)/4294967296.0
            radius=math.sqrt(-2.0*math.log(u1)); angle=2.0*math.pi*u2
            result.append(radius*math.cos(angle)); result.append(radius*math.sin(angle)); spare=result[-1]
        del result[count:]
        return result.tobytes(),struct.pack('<QQfi',self.state,self.increment,spare,count%2)

def verify(path):
    parts={p[0]:p[-1] for p in entries(Path(path).read_bytes())}
    # Generation identity has four leading words and 29 ordered parameter words.
    seed=struct.unpack_from('<Q',parts[1],16+29*4)[0]
    counts=[len(parts[12])//4,len(parts[13])//4]
    assert struct.unpack_from('<I',parts[14])[0]==1
    assert struct.unpack_from('<2Q',parts[14],52)==tuple(counts)
    for index,(section,count) in enumerate(zip([21,22],counts)):
        noise,rng=PCG(seed).normals(count)
        assert noise==parts[section],f'{path}: original noise differs in section {section}'
        assert rng==parts[14][4+24*index:28+24*index],f'{path}: current RNG state differs'
    print(f'ok: independent original-noise/RNG oracle {path}, {sum(counts)} normals')

if __name__=='__main__':
    assert sys.byteorder=='little'
    if len(sys.argv)<2: raise SystemExit('usage: sampler_noise.py CHECKPOINT ...')
    for path in sys.argv[1:]: verify(path)
