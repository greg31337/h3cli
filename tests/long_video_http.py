"""Fixed offline HTTP native/still regression, invoked by the bounded runner."""
import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import shlex
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reject-only',action='store_true')
    ap.add_argument('--precision',choices=('bf16','nvfp4'),default='bf16')
    ap.add_argument('--attention',choices=('default','sage3'),default='default')
    ap.add_argument('--quant-cache')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    def request(method,path,body=None):
        conn=http.client.HTTPConnection('127.0.0.1',port,timeout=30)
        conn.request(method,path,json.dumps(body) if body else None,{'Content-Type':'application/json'})
        response=conn.getresponse();data=json.loads(response.read());conn.close()
        return response.status,data
    records=[]
    with (a.out/'server.log').open('w') as log:
        server=subprocess.Popen(['bin/h3cli','-d',a.model,'--offline','--server',
            '--server-port',str(port),'--server-state-dir',str(a.out/'state')],
            env=dict(os.environ,H3_OFFLINE='1'),stdout=log,stderr=log)
        try:
            for _ in range(100):
                try:
                    if request('GET','/health')[0]==200:break
                except OSError:pass
                assert server.poll() is None;time.sleep(.1)
            status,error=request('POST','/v1/h3/jobs',{'h3cli':'--width 256 --height 256 --frames 1451 --steps 2 -p "A teapot."'})
            assert status in (400,422),(status,error)
            if a.reject_only:
                if sys.platform=='darwin':
                    status,error=request('POST','/v1/h3/jobs',{'h3cli':'--width 256 --height 256 --frames 464 --steps 2 -p "A teapot."'})
                    assert status in (400,422),(status,error)
                (a.out/'result.json').write_text(json.dumps(dict(status='passed',rejection_only=True))+'\n')
                return
            for label,flags,size,frames in (
                ('video','--width 256 --height 256 --frames 464',(256,256),464),
                ('still','--still --width 256 --height 256',(256,256),1)):
                if label=='video' and a.precision=='nvfp4':
                    flags+=' --cuda-denoise-quant nvfp4'
                    if a.quant_cache:flags+=' --cuda-denoise-quant-cache '+shlex.quote(a.quant_cache)
                if label=='video' and a.attention!='default':flags+=' --cuda-attention '+a.attention
                if label=='video':flags+=' --save-av-state result.h3av'
                status,job=request('POST','/v1/h3/jobs',{'h3cli':flags+' --steps 2 --seed 42 -p "A red ceramic teapot on a wooden table."'})
                assert status==202,(status,job)
                end=time.monotonic()+1200
                while time.monotonic()<end:
                    status,job=request('GET','/v1/h3/jobs/'+job['id'])
                    assert status==200
                    if job['status'] not in ('queued','running'):break
                    time.sleep(.5)
                assert job['status']=='completed',job
                results=list((a.out/'state/jobs'/job['id']).rglob('completion.json'));assert results
                for path in results:
                    completion=json.loads(path.read_text())
                    assert (completion['width'],completion['height'],completion['frames'])==(*size,frames)
                if label=='video':
                    sidecars=list((a.out/'state/jobs'/job['id']).rglob('*.h3av.presentation'))
                    assert sidecars
                    for path in sidecars:
                        fields=dict(line.split(' ',1) for line in path.read_text().splitlines())
                        assert int(fields['attention'].split()[0])==(2 if a.attention=='sage3' else 0)
                        assert int(fields['denoise_quant'].split()[0])==(2 if a.precision=='nvfp4' else 0)
                records.append(dict(kind=label,job=job))
            (a.out/'result.json').write_text(json.dumps(records,indent=2)+'\n')
        finally:
            server.terminate();server.wait(timeout=30)
    print('PASS HTTP extended native video, early geometry rejection, legacy still')


if __name__=='__main__':main()
