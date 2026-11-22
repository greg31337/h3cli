#!/usr/bin/env python3
"""Real-model stock still generation, repeatability, and codec-only qualification."""
import argparse, hashlib, json, os, struct, subprocess, time
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--high',action='store_true')
    parser.add_argument('--quality',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    prompt='A polished red ceramic teapot on a pale wooden table, a single still photograph, soft studio lighting.'
    reference='A photograph of the person in image 1, wearing a blue jacket, standing in a sunlit garden.'
    refs=['tests/fixtures/cuda-reference/portrait.jpg','tests/fixtures/cuda-reference/first.png']
    cases=[('text-640',640,480,2,'match',[]),('text-640-repeat',640,480,2,'match',[]),
           ('text-1344',1344,768,2,'match',[]),('ref-640-match',640,480,2,'match',refs),
           ('ref-640-high',640,480,2,'high',refs),('ref-640-max',640,480,2,'max',refs),
           ('ref-640-repeat',640,480,2,'match',refs),('ref-640-reversed',640,480,2,'match',refs[::-1]),
           ('ref-1344',1344,768,2,'high',refs)]
    if args.high:cases=[('text-2144',2144,1216,2,'match',[]),('ref-2144',2144,1216,2,'high',refs[:1]),
                       ('text-8mp',3872,2176,2,'match',[]),('ref-8mp',3872,2176,2,'high',refs[:1]),
                       ('portrait-8mp',2176,3872,2,'high',refs[:1])]
    if args.quality:cases=[('quality-text-8mp',3872,2176,50,'match',[]),('quality-ref-8mp',3872,2176,50,'high',refs[:1])]
    env=dict(os.environ,H3_OFFLINE='1',H3_PROFILE='1',H3_VERBOSE='1')
    env.pop('H3_TEST_MAX_EVALUATIONS',None)
    records=[]
    def save():
        (args.out/'result.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in records),cases=records,
            binary_sha256=sha('bin/h3cli'),probe_sha256=sha('bin/still_generate')),indent=2)+'\n')
    for name,w,h,steps,mode,images in cases:
        folder=args.out/name;folder.mkdir()
        command=['bin/still_generate',str(args.model),str(folder),str(w),str(h),str(steps),mode,reference if images else prompt,*images]
        begin=time.monotonic()
        with (folder/'run.log').open('w') as log,(folder/'metrics.json').open('w') as metrics:
            result=subprocess.run(command,env=env,stdout=metrics,stderr=log)
        record=dict(name=name,command=command,width=w,height=h,steps=steps,seed=42,reference_size=mode,seconds=time.monotonic()-begin,returncode=result.returncode,passed=False)
        records.append(record)
        if result.returncode==0:
            metrics=json.loads((folder/'metrics.json').read_text());record.update(metrics)
            raw=(folder/'output.png').read_bytes();assert raw[:8]==b'\x89PNG\r\n\x1a\n'
            assert struct.unpack('>II',raw[16:24])==(w,h)
            latent=folder/'latent.safetensors';data=latent.read_bytes();n=struct.unpack('<Q',data[:8])[0];header=json.loads(data[8:8+n]);assert header['latent']['shape']==[1,24,1,h//16,w//16]
            record['png_sha256']=sha(folder/'output.png');record['latent_sha256']=sha(latent)
            decode=['bin/h3cli','-d',str(args.model),'--offline','--decode-still-latent',str(latent),'-o',str(folder/'decoded.png')]
            with (folder/'decode.log').open('w') as log:result=subprocess.run(decode,env=env,stdout=log,stderr=subprocess.STDOUT)
            record['decode_command']=decode
            record['passed']=metrics['passed'] and result.returncode==0 and sha(folder/'decoded.png')==record['png_sha256']
        save();print(name,record['passed'],round(record['seconds'],2),flush=True)
        if not record['passed']:return 1
    by={r['name']:r for r in records}
    for a,b in [('text-640','text-640-repeat'),('ref-640-match','ref-640-repeat')]:
        if a in by:
            for key in ('trajectory_sha256','latent_sha256','png_sha256'):assert by[a][key]==by[b][key],(a,b,key)
    if 'ref-640-reversed' in by:assert by['ref-640-match']['latent_sha256']!=by['ref-640-reversed']['latent_sha256']
    save();return 0
if __name__=='__main__':raise SystemExit(main())
