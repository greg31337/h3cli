#!/usr/bin/env python3
"""Smoke-test the offline gallery in an installed Chrome (websocket-client)."""
import argparse,base64,hashlib,json,pathlib,socket,subprocess,tempfile,time,urllib.request
import websocket

def main():
    p=argparse.ArgumentParser();p.add_argument('page');p.add_argument('--chrome',default='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');a=p.parse_args();page=pathlib.Path(a.page).resolve()
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='h3-gallery-browser-') as profile, (page.parent/'browser.log').open('w') as log:
        process=subprocess.Popen([a.chrome,'--headless=new','--disable-gpu','--no-first-run','--disable-background-networking','--autoplay-policy=no-user-gesture-required',f'--user-data-dir={profile}',f'--remote-debugging-port={port}','--remote-debugging-address=127.0.0.1',page.as_uri()],stdout=log,stderr=log)
        try:
            target=None
            for _ in range(100):
                try:
                    targets=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list',timeout=1));target=next((x for x in targets if x.get('url')==page.as_uri()),None)
                    if target:break
                except (OSError,ValueError):pass
                time.sleep(.1)
            assert target,'Chrome did not open report';ws=websocket.create_connection(target['webSocketDebuggerUrl'],timeout=20,suppress_origin=True);counter=0;errors=[]
            def call(method,params=None):
                nonlocal counter
                counter+=1;ws.send(json.dumps(dict(id=counter,method=method,params=params or {})))
                while True:
                    message=json.loads(ws.recv())
                    if message.get('method')=='Runtime.exceptionThrown':errors.append(message)
                    if message.get('id')==counter:
                        assert 'error' not in message,message
                        return message.get('result',{})
            def evaluate(expression):
                response=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True));assert 'exceptionDetails' not in response,response
                return response.get('result',{}).get('value')
            call('Runtime.enable');call('Page.enable')
            for _ in range(100):
                ready=evaluate("(()=>{let v=document.querySelector('.comparison video');return v&&v.readyState>=1})()")
                if ready:break
                time.sleep(.1)
            assert ready,'video metadata unavailable'
            setup=evaluate("(()=>{window.testCard=document.querySelector('.comparison');window.testVideos=[...testCard.querySelectorAll('video')];testCard.scrollIntoView();return {players:testVideos.length,controls:!!testCard.querySelector('input'),duration:testVideos[0].duration}})()")
            assert setup['players']>=3 and setup['controls'] and setup['duration']>0,setup
            evaluate("testCard.querySelector('button').click()");time.sleep(.8)
            playing=evaluate("testVideos.map(v=>({paused:v.paused,time:v.currentTime,error:v.error&&v.error.code}))")
            assert all(not x['paused'] and x['time']>0 and not x['error'] for x in playing),playing
            evaluate("testVideos.forEach(v=>v.pause());let slider=testCard.querySelector('input');slider.value='0.5';slider.dispatchEvent(new Event('input'))")
            time.sleep(.5);seek=evaluate('testVideos.map(v=>v.currentTime)');assert max(abs(x-.5) for x in seek)<.12,seek
            evaluate('testVideos[0].currentTime=1.0');time.sleep(.5);native=evaluate('testVideos.map(v=>v.currentTime)');assert max(abs(x-1.) for x in native)<.12,native
            screenshot=call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))
            (page.parent/'browser.png').write_bytes(base64.b64decode(screenshot['data']));assert not errors,errors
            result=dict(page=str(page),harness_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),setup=setup,playing=playing,shared_seek=seek,native_seek=native,javascript_exceptions=errors,passed=True)
            (page.parent/'browser-validation.json').write_text(json.dumps(result,indent=2)+'\n');ws.close();print('PASS offline media, shared play/pause/seek, native seek propagation, no JavaScript exceptions')
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
if __name__=='__main__':main()
