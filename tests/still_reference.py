#!/usr/bin/env python3
"""Independent F32 PyTorch T=1 VAE equations; never calls the h3cli decoder.
Equations: pinned MiniMax H3 vae_vit/base_module/attention/conv/vae_cnn.
The T=1 causal zero padding collapses temporal Conv3D to its last K_t Conv2D.
All slice projections and unclamped RGB are retained. GPU jobs run serially.
"""
import argparse, hashlib, json, math, struct, subprocess, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while x:=f.read(8<<20):h.update(x)
    return h.hexdigest()
class Weights:
    def __init__(self,path,device):
        self.path=path;self.device=device
        with open(path,'rb') as f:
            self.start=8+struct.unpack('<Q',f.read(8))[0];self.header=json.loads(f.read(self.start-8))
        self.config=json.loads(self.header['__metadata__']['minimax_h3_video_vae'])
        self.cache={}
    def __call__(self,key):
        if key not in self.cache:
            t=self.header[key]
            a=np.memmap(self.path,dtype={'F16':'<f2','F32':'<f4'}[t['dtype']],mode='r',offset=self.start+t['data_offsets'][0],shape=tuple(t['shape']))
            self.cache[key]=torch.from_numpy(np.array(a,dtype=np.float32)).to(self.device)
        return self.cache[key]

def axis(extent,tile=256):
    if extent<=tile:return [0],extent,[]
    n=math.ceil(extent/tile)
    while n*tile-(n-1)*64<extent:n+=1
    overlaps=[64]*(n-1);remaining=n*tile-(n-1)*64-extent
    i=0
    while remaining:overlaps[i%(n-1)]+=16;remaining-=16;i+=1
    starts=[0]
    for o in overlaps:starts.append(starts[-1]+tile-o)
    return starts,tile,overlaps

def assemble(tiles,ys,xs,th,tw,oy,ox):
    # Released row-major pairwise blending: raw neighbor tiles, then crop tails.
    out=np.empty((*tiles[0][0].shape[:-3],ys[-1]+th,xs[-1]+tw,3),np.float32)
    for j,y in enumerate(ys):
        for i,x in enumerate(xs):
            kh=th-(oy[j] if j<len(ys)-1 else 0);kw=tw-(ox[i] if i<len(xs)-1 else 0)
            a=tiles[j][i][...,:kh,:kw,:].copy()
            if j:
                n=min(oy[j-1],kh);alpha=(np.arange(n,dtype=np.float32)/oy[j-1]).reshape((-1,1,1))
                a[...,:n,:,:]=tiles[j-1][i][...,th-oy[j-1]:th-oy[j-1]+n,:kw,:]*(1-alpha)+a[...,:n,:,:]*alpha
            if i:
                n=min(ox[i-1],kw);alpha=(np.arange(n,dtype=np.float32)/ox[i-1]).reshape((1,-1,1))
                a[...,:,:n,:]=tiles[j][i-1][...,:kh,tw-ox[i-1]:tw-ox[i-1]+n,:]*(1-alpha)+a[...,:,:n,:]*alpha
            out[...,y:y+kh,x:x+kw,:]=a
    return out

def rope(h,w,device):
    coords=torch.stack(torch.meshgrid(torch.zeros(1),2*(torch.arange(h)+.5)/h-1,2*(torch.arange(w)+.5)/w-1,indexing='ij'),-1).reshape(-1,3)
    coords=torch.cat([coords,torch.zeros(5,3)],0).to(device)
    angle=(2*math.pi*coords[:,:,None]*(100**(-torch.arange(8,device=device)/8))).reshape(-1,24)
    return angle.cos(),angle.sin()

def dense_decode(w,z,dest=None):
    _,h,ww=z.shape
    x=torch.from_numpy(z.transpose(1,2,0).copy()).to(w.device).reshape(-1,24)
    x=F.linear(x,w('post_quant_conv.weight').reshape(24,24),w('post_quant_conv.bias'))
    x=F.linear(x,w('decoder.x_embedder.weight'),w('decoder.x_embedder.bias'))
    x=torch.cat([x,w('decoder.register_tokens')[0],torch.zeros((1,2048),device=w.device)])
    c,s=rope(h,ww,w.device)
    if dest:np.savez(dest/'rope.npz',cos=c.cpu().numpy(),sin=s.cpu().numpy())
    def rms(a,weight=None):
        a=a*torch.rsqrt(a.square().mean(-1,keepdim=True)+1e-5)
        return a if weight is None else a*weight
    def rot(a):
        return torch.cat([a[...,:24]*c[:,None]-a[...,24:48]*s[:,None],a[...,24:48]*c[:,None]+a[...,:24]*s[:,None],a[...,48:]],-1)
    for i in range(36):
        p=f'decoder.transformer_blocks.{i}.'
        q,k,v=F.linear(rms(x,w(p+'norm1.weight')),w(p+'attn.to_qkv.weight'),w(p+'attn.to_qkv.bias')).reshape(-1,32,192).chunk(3,-1)
        q,k=rot(rms(q)),rot(rms(k))
        # Explicit softmax/matmul reference, independent of the candidate SDPA.
        a=torch.softmax((q.transpose(0,1)@k.permute(1,2,0))/8,dim=-1)@v.transpose(0,1)
        x=x+F.linear(a.transpose(0,1).reshape(-1,2048),w(p+'attn.to_out.weight'),w(p+'attn.to_out.bias'))*w(p+'scale1')
        gate,up=F.linear(rms(x,w(p+'norm2.weight')),w(p+'ff.w1.weight'),w(p+'ff.w1.bias')).chunk(2,-1)
        x=x+F.linear(F.silu(gate)*up,w(p+'ff.w2.weight'),w(p+'ff.w2.bias'))*w(p+'scale2')
    x=F.layer_norm(x,(2048,),w('decoder.norm_out.weight'),w('decoder.norm_out.bias'),1e-5)
    p=F.linear(x,w('decoder.proj_out.weight'),w('decoder.proj_out.bias'))[:h*ww].cpu().numpy()
    if dest:np.save(dest/'projection.npy',p)
    rgb=p.reshape(h,ww,3,4,16,16).transpose(3,0,4,1,5,2).reshape(4,h*16,ww*16,3)
    return rgb*np.array([.229,.224,.225],np.float32)+np.array([.485,.456,.406],np.float32)

def decode(w,z,tile=256,dest=None):
    mean=np.array(w.config['latents_mean'],np.float32)[:,None,None];std=np.array(w.config['latents_std'],np.float32)[:,None,None]
    raw=z*std+mean
    if dest:np.save(dest/'raw_latent.npy',raw)
    ys,th,oy=axis(z.shape[1]*16,tile);xs,tw,ox=axis(z.shape[2]*16,tile)
    tiles=[]
    for j,y in enumerate(ys):
        row=[]
        for i,x in enumerate(xs):
            row.append(dense_decode(w,raw[:,y//16:(y+th)//16,x//16:(x+tw)//16],dest if not i and not j else None))
        tiles.append(row)
    return assemble(tiles,ys,xs,th,tw,oy,ox)

def encode_tile(w,rgb):
    x=torch.from_numpy(rgb.transpose(2,0,1).copy()[None]).to(w.device)
    x=(x-torch.tensor([.485,.456,.406],device=w.device)[None,:,None,None])/torch.tensor([.229,.224,.225],device=w.device)[None,:,None,None]
    def conv(x,name,stride=1,pad=(1,1,1,1)):
        k=w(name+'.weight')[:,:,-1];x=F.pad(x,pad,mode='reflect') if any(pad) else x
        return F.conv2d(x,k,w(name+'.bias'),stride=stride)
    def norm(x,name):return F.silu(F.group_norm(x,32,w(name+'.weight'),w(name+'.bias'),1e-6))
    x=conv(x,'encoder.conv_in')
    for level in range(6):
        for b in range(2):
            p=f'encoder.down.{level}.block.{b}';old=x
            x=conv(norm(x,p+'.norm1'),p+'.conv1');x=conv(norm(x,p+'.norm2'),p+'.conv2')
            if p+'.nin_shortcut.weight' in w.header:old=conv(old,p+'.nin_shortcut',pad=(0,0,0,0))
            x=x+old
        p=f'encoder.down.{level}.downsample.conv'
        if p+'.weight' in w.header:x=conv(x,p,stride=2,pad=(0,1,0,1))
    x=conv(norm(x,'encoder.norm_out'),'encoder.conv_out')
    return conv(x,'quant_conv',pad=(0,0,0,0))[0,:24].cpu().numpy()

def encode(w,rgb):
    h,ww,_=rgb.shape;ys,th,oy=axis(h);xs,tw,ox=axis(ww)
    # Reuse spatial stitching for arbitrary channels via a separate channel loop.
    tiles=[[encode_tile(w,rgb[y:y+th,x:x+tw]) for x in xs] for y in ys]
    result=np.empty((24,h//16,ww//16),np.float32)
    for c in range(24):
        rgb_tiles=[[np.repeat(t[c,:,:,None],3,-1) for t in row] for row in tiles]
        result[c]=assemble(rgb_tiles,[x//16 for x in ys],[x//16 for x in xs],th//16,tw//16,[x//16 for x in oy],[x//16 for x in ox])[...,0]
    return (result-np.array(w.config['latents_mean'],np.float32)[:,None,None])/np.array(w.config['latents_std'],np.float32)[:,None,None]

def load_latent(path):
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));t=h['latent'];f.seek(8+n+t['data_offsets'][0]);return np.frombuffer(f.read(),'<f4').reshape(t['shape'])[0,:,0].copy()

def crop(path,w,h):
    cmd=['ffmpeg','-v','error','-i',str(path),'-vf',f'scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}', '-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    return np.frombuffer(subprocess.check_output(cmd),np.uint8).reshape(h,w,3).astype(np.float32)/255

def main():
    p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('fixture',type=Path);p.add_argument('--device',default='mps');p.add_argument('--encoder',action='store_true');p.add_argument('--whole',action='store_true');a=p.parse_args()
    torch.set_num_threads(8);torch.set_grad_enabled(False);start=time.monotonic();w=Weights(a.model,a.device)
    if a.device.startswith('cuda'):
        # Keep the independent F32 oracle out of TF32 convolution/matmul paths.
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    z=load_latent(a.fixture/'latent.safetensors');source=crop('inputs/2.jpg',z.shape[2]*16,z.shape[1]*16)
    source.tofile(a.fixture/'source.f32')
    if a.encoder:
        ref=encode(w,source);np.save(a.fixture/'reference_latent.npy',ref);print('encoder MAE',np.abs(ref-z).mean(),flush=True)
        w.cache.clear()
    trace=a.fixture/'whole' if a.whole else a.fixture
    trace.mkdir(exist_ok=True)
    rgb=decode(w,z,tile=max(z.shape[1:])*16 if a.whole else 256,dest=trace)
    np.save(a.fixture/('whole_slices.npy' if a.whole else 'reference_slices.npy'),rgb)
    report={'seconds':time.monotonic()-start,'device':a.device,'dtype':'F32','model_sha256':sha(a.model),'latent_sha256':sha(a.fixture/'latent.safetensors'),'script_sha256':sha(__file__),'contract_sha256':sha('tests/still_contract.json')}
    if a.device.startswith('cuda'):report.update(tf32=False,torch_version=torch.__version__,gpu=torch.cuda.get_device_name())
    (a.fixture/('whole-reference.json' if a.whole else 'reference.json')).write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
