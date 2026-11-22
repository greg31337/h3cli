#!/usr/bin/env python3
"""Exercise a real native HTTP still job with provisioned offline weights."""
import argparse,http.client,json,os,socket,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
def request(method,path,body=None):
    conn=http.client.HTTPConnection('127.0.0.1',port,timeout=30)
    conn.request(method,path,json.dumps(body) if body else None,{'Content-Type':'application/json'})
    response=conn.getresponse();data=json.loads(response.read());conn.close();return response.status,data
with (a.out/'server.log').open('w') as log:
    server=subprocess.Popen(['bin/h3cli','-d',a.model,'--offline','--server','--server-port',str(port),'--server-state-dir',str(a.out/'state')],env=dict(os.environ,H3_OFFLINE='1'),stdout=log,stderr=log)
    try:
        for _ in range(100):
            try:
                if request('GET','/health')[0]==200:break
            except OSError:pass
            assert server.poll() is None;time.sleep(.1)
        status,job=request('POST','/v1/h3/jobs',{'h3cli':'--still --width 256 --height 256 --steps 2 --seed 42 -p "A red ceramic teapot on a wooden table."'})
        assert status==202,(status,job)
        end=time.monotonic()+600
        while time.monotonic()<end:
            status,job=request('GET','/v1/h3/jobs/'+job['id'])
            assert status==200
            if job['status'] not in ('queued','running'):break
            time.sleep(.2)
        (a.out/'result.json').write_text(json.dumps(job,indent=2)+'\n')
        assert job['status']=='completed',job
        results=list((a.out/'state/jobs'/job['id']).rglob('completion.json'))
        assert results
        metrics=[json.loads(file.read_text()) for file in results]
        for result in metrics:
            assert result['kind']=='still' and result['frames']==1 and result['fps']==0
            assert result['audio_samples']==0 and result['sample_rate']==0
        artifacts=job['h3']['artifacts'];assert any(item['name']=='image.png' for item in artifacts)
        assert not any(item['name'].endswith(('.h3av','.wav','.mp4')) for item in artifacts)
        print('PASS HTTP result: one PNG, still kind, one frame, zero audio and AV state')
    finally:
        server.terminate();server.wait(timeout=30)
