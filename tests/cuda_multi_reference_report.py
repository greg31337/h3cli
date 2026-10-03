#!/usr/bin/env python3
"""GPU-free reports, galleries, validation audit and reproducible bug ledger."""
import csv
import html
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from cuda_multi_reference import ROOT, compare, records, save, sha, load_manifest


def e(value): return html.escape(str(value))
def number(value, unit=''): return 'N/A' if value is None else f'{value:.2f}{unit}'


def black_frame_check(movie):
    """Flag sustained nearly-black output; this is a visual heuristic only."""
    path=movie.parent/'black-frame-check.json'
    checksum=sha(movie)
    if path.exists():
        result=json.loads(path.read_text())
        if result.get('method')=='ffmpeg-blackdetect-v1' and result['input_sha256']==checksum:
            return result
    log=movie.parent/'black-frame-check.log'
    args=['ffmpeg','-hide_banner','-nostats','-threads','2','-i',str(movie),
          '-map','0:v:0','-filter_threads','2','-vf',
          'blackdetect=d=0.5:pix_th=0.02:pic_th=0.98','-an','-f','null','-']
    with log.open('w') as stream:
        subprocess.run(args,stdout=subprocess.DEVNULL,stderr=stream,check=True)
    spans=[dict(zip(('start_seconds','end_seconds','duration_seconds'),map(float,values)))
           for values in re.findall(r'black_start:([\d.]+)\s+black_end:([\d.]+)\s+black_duration:([\d.]+)',log.read_text())]
    result={'method':'ffmpeg-blackdetect-v1','input_sha256':checksum,'argv':args,
            'minimum_duration_seconds':.5,'minimum_black_pixel_fraction':.98,
            'pixel_luma_threshold_fraction':.02,'intervals':spans,'warning':bool(spans),
            'log':str(log.relative_to(ROOT))}
    save(path,result)
    return result


def summarize_bug(r):
    logpath=ROOT/r['artifact_directory']/'stderr.log'
    log=logpath.read_text(errors='replace') if logpath.exists() else ''
    reason=r.get('termination_reason') or r.get('block_reason')
    if 'memory safety:' in log: return 'MR-B001','resource','Renderer memory guard abort',log.split('memory safety:')[-1][-1200:]
    if re.search(r'out of memory|cudaErrorMemoryAllocation',log,re.I): return 'MR-B002','product','CUDA/host allocation failure',log[-1600:]
    if reason: return 'MR-B003','environment','External resource/telemetry/timeout stop',reason
    lines=[s.strip() for s in log.replace('\r','\n').splitlines() if s.strip().startswith('h3cli:')]
    if r.get('returncode'):
        message=lines[-1] if lines else log[-600:]
        key=__import__('hashlib').sha256(message.encode()).hexdigest()[:8]
        return 'MR-E-'+key,'product','Generation failed',message
    if r.get('completed_steps')!=r['case']['steps']:
        return 'MR-H001','harness','Step completion record missing or inconsistent',str(r.get('completed_steps'))
    return 'MR-B004','product','Output media validation failed',json.dumps(r.get('validation',{}).get('errors',[]))


def visual_artifacts(r):
    if not r.get('output'): return
    movie=ROOT/r['output']
    directory=movie.parent
    info=r.get('validation',{}).get('probe',{})
    streams=info.get('streams',[])
    video=next((s for s in streams if s.get('codec_type')=='video'),None)
    if not video: return
    frames=int(video.get('nb_read_frames',0))
    if frames<6: return
    contact=directory/'contact.jpg'
    if not contact.exists():
        indices=sorted({round(i*(frames-1)/5) for i in range(6)})
        select='+'.join(f'eq(n\\,{n})' for n in indices)
        filters=f'select={select},scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2,setsar=1,tile=3x2'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(movie),'-vf',filters,
                        '-frames:v','1',str(contact)],check=True)
    r['contact_sheet']=str(contact.relative_to(ROOT))
    hashes=directory/'frames.md5'
    if not hashes.exists():
        with hashes.open('w') as f:
            subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-map','0:v:0',
                            '-f','framemd5','-'],stdout=f,check=True)
    h=[line.split(',')[-1].strip() for line in hashes.read_text().splitlines() if line and not line.startswith('#')]
    r['decoded_frame_identity']={'frames':len(h),'unique_frames':len(set(h)),
                                 'frozen_warning':len(set(h))<max(2,len(h)*.05)}


def report():
    m=load_manifest()
    environment=json.loads((ROOT/'environment.json').read_text())
    rows=records()
    all_rows=records(include_superseded=True)
    active_directories={r['artifact_directory'] for r in rows}
    superseded=[r for r in all_rows if r['artifact_directory'] not in active_directories]
    retries_path=ROOT/'retries.json'
    retry_decisions=json.loads(retries_path.read_text()) if retries_path.exists() else []
    by_id={r['case']['id']:r for r in rows}
    scheduled={c['id']:c for c in m['cases']}
    assert len(by_id)==len(rows), 'Multiple attempts need explicit supersession before reporting'
    assert all(r['manifest_identity']==m['identity'] for r in rows),'Stale evidence'
    assert all(r['case']==scheduled[r['case']['id']] for r in rows),'Case definition changed'
    assert all(r.get('final_identity')==m.get('final_identity') for r in rows
               if r['case']['phase']=='main'),'Final evidence belongs to another configuration'
    assert all(r.get('sampling_identity')==m.get('sampling_identity') for r in rows
               if r['case']['phase']=='sample'),'Sample evidence belongs to another configuration'
    blocked_path=ROOT/'blocked.json'
    blocked=json.loads(blocked_path.read_text()) if blocked_path.exists() else {}
    bugs={}
    corrections_path=ROOT/'harness-corrections.json'
    corrections=json.loads(corrections_path.read_text()) if corrections_path.exists() else []
    observations_path=ROOT/'visual-observations.json'
    observations=json.loads(observations_path.read_text()) if observations_path.exists() else []
    interruptions_path=ROOT/'interruptions.json'
    interruptions=json.loads(interruptions_path.read_text()) if interruptions_path.exists() else {}
    for r in rows:
        if r['case']['id'] in interruptions:
            r['status']='interrupted'
            r['interruption']=interruptions[r['case']['id']]
        r['comparison']=compare(r,rows)
        idle=r.get('telemetry',{}).get('idle_bytes')
        r['idle_vram_gib']=idle/(1<<30) if idle is not None else None
        r['bug_ids']=[]
        if r.get('status') not in ('complete','interrupted'):
            key,category,title,detail=summarize_bug(r)
            b=bugs.setdefault(key,{'id':key,'category':category,'title':title,
                'severity':'blocks affected test','fix_status':'deferred by user',
                'expected':'Complete generation with all supplied references within the frozen limits',
                'cases':[],'evidence':[]})
            b['cases'].append(r['case']['id'])
            b['evidence'].append({'case':r['case']['id'],'observed':detail,'argv':r.get('argv'),
                                  'request':r['artifact_directory']+'/request.json',
                                  'stderr':r['artifact_directory']+'/stderr.log',
                                  'telemetry':r['artifact_directory']+'/telemetry.jsonl.gz'})
            r['bug_ids'].append(key)
        expected=[]
        image_geometry=[]
        for i,k in enumerate(r['case']['references'],1):
            a=m['assets'][k]
            if a['kind']=='video':
                expected.append({'reference_index':i,'pipeline':'released-v1',
                    'normalized_frames':a['frames'],'vae_frames':a['vae_frames'],
                    'latent_t':a['latent_t'],'posterior_seed':'42'})
            else:
                c=r['case']
                width,height=a['canvases'][f'{c["width"]}x{c["height"]}-{c["image_size"]}']
                image_geometry.append({'reference_index':i,'asset':k,
                    'canvas_wh':[width,height],'vae_latent_thw':[1,height//16,width//16],
                    'conditioning_rows':(height//32)*(width//32),
                    'source':'host-derived from h3_reference_image_canvas, h3_latent_canvas and the 2x2 conditioning patch layout; not a runtime diagnostic'})
        observed=r.get('video_preprocessing',[])
        logpath=ROOT/r['artifact_directory']/'stderr.log'
        log=logpath.read_text(errors='replace') if logpath.exists() else ''
        vision_counts=re.findall(r'reference vision preparation\s+(\d+)\s*/\s*(\d+)',log)
        audio_counts=re.findall(r'audio VAE encoder\s+(\d+)\s*/\s*(\d+)',log)
        expected_vision=sum(1 if m['assets'][k]['kind']=='image' else
            (math.ceil(m['assets'][k]['frames']/12)+1)//2 for k in r['case']['references'])
        r['reference_audit']={'expected_videos':expected,'observed_videos':observed,
                              'all_video_geometry_logged':expected==observed,
                              'expected_vision_blocks':expected_vision,
                              'observed_vision_completion':vision_counts[-1] if vision_counts else None,
                              'all_vision_blocks_logged':
                                  (not r['case']['references'] or bool(vision_counts) and
                                   tuple(map(int,vision_counts[-1]))==(expected_vision,expected_vision)),
                              'embedded_audio_requested':r['case']['embedded_audio'],
                              'audio_encoder_completion':audio_counts[-1] if audio_counts else None,
                              'audio_encoder_matches_request':bool(audio_counts and
                                  audio_counts[-1][0]==audio_counts[-1][1])==r['case']['embedded_audio'],
                              'image_geometry':image_geometry,
                              'image_canvases':'host-derived; per-image runtime diagnostics unavailable'}
        if r.get('status')=='complete' and (expected!=observed or
                not r['reference_audit']['all_vision_blocks_logged'] or
                not r['reference_audit']['audio_encoder_matches_request']):
            key='MR-B005'
            bugs.setdefault(key,{'id':key,'category':'product','title':'Reference preprocessing audit mismatch',
                'severity':'invalidates reference coverage','fix_status':'deferred by user','cases':[],
                'expected':'Every video has the planned geometry, all vision blocks complete, and the audio encoder matches the embedded-audio policy','evidence':[]})['cases'].append(r['case']['id'])
            bugs[key]['evidence'].append({'case':r['case']['id'],
                'observed':json.dumps(r['reference_audit']),
                'request':r['artifact_directory']+'/request.json',
                'stderr':r['artifact_directory']+'/stderr.log',
                'telemetry':r['artifact_directory']+'/telemetry.jsonl.gz'})
            r['bug_ids'].append(key)
        visual_artifacts(r)
        if r.get('output') and r.get('validation',{}).get('valid'):
            r['black_frame_check']=black_frame_check(ROOT/r['output'])
            if r['black_frame_check']['warning']:
                key='MR-V002'
                bugs.setdefault(key,{'id':key,'category':'visual','title':'Sustained nearly-black output',
                    'severity':'heuristic warning; visual inspection required','fix_status':'deferred by user',
                    'cases':[],'expected':'No sustained nearly-black span in the requested daylight scene',
                    'evidence':[]})['cases'].append(r['case']['id'])
                bugs[key]['evidence'].append({'case':r['case']['id'],
                    'observed':json.dumps(r['black_frame_check']['intervals']),
                    'request':r['artifact_directory']+'/request.json',
                    'stderr':r['black_frame_check']['log'],
                    'telemetry':r['artifact_directory']+'/telemetry.jsonl.gz'})
                r['bug_ids'].append(key)
        if r.get('decoded_frame_identity',{}).get('frozen_warning'):
            key='MR-V001'
            bugs.setdefault(key,{'id':key,'category':'visual','title':'Nearly frozen decoded output',
                'severity':'suspected; visual inspection required','fix_status':'deferred by user',
                'cases':[],'expected':'Visible temporal variation in the requested moving scene',
                'evidence':[]})['cases'].append(r['case']['id'])
            r['bug_ids'].append(key)
        r['visual_observations']=[o for o in observations if o['case']==r['case']['id']]
    repeats=[]
    for r in rows:
        original=by_id.get(r['case'].get('repeat_of'))
        if original and original['status']=='complete' and r['status']=='complete':
            def pixels(row):
                path=ROOT/row['artifact_directory']/'frames.md5'
                # FFmpeg-version comments are not decoded image content.
                return [line.split(',')[-1].strip() for line in path.read_text().splitlines()
                        if line and not line.startswith('#')]
            repeats.append({'case':r['case']['id'],'original':original['case']['id'],
                'identical_mp4':r['validation']['sha256']==original['validation']['sha256'],
                'identical_decoded_video':pixels(original)==pixels(r),
                'wall_delta_seconds':r['wall_seconds']-original['wall_seconds'],
                'peak_vram_delta_gib':r['max_vram_gib']-original['max_vram_gib'],
                'gpu_released':r['gpu_released'] and original['gpu_released']})
    audio_comparisons=[]
    for width in (640,1344):
        for audio,silent in [('A01','V01'),('A11','X11-match')]:
            a,b=by_id.get(f'R{width}-{audio}'),by_id.get(f'R{width}-{silent}')
            if not a or not b or a['status']!='complete' or b['status']!='complete':continue
            audio_comparisons.append({'case':a['case']['id'],'silent_case':b['case']['id'],
                'wall_delta_seconds':a['wall_seconds']-b['wall_seconds'],
                'wall_ratio':a['wall_seconds']/b['wall_seconds'],
                'peak_vram_delta_gib':a['max_vram_gib']-b['max_vram_gib'],
                'audio_encoder_completion':a['reference_audit']['audio_encoder_completion']})
    for case,reason in blocked.items():
        assert case in scheduled and case not in by_id
    absent=[c for k,c in scheduled.items() if k not in by_id and k not in blocked]
    coverage={'scheduled':len(scheduled),'measured':len(rows),
              'complete':sum(r['status']=='complete' for r in rows),
              'failed_or_blocked_attempts':sum(r['status'] not in ('complete','interrupted') for r in rows),
              'interrupted_by_user':sum(r['status']=='interrupted' for r in rows),
              'superseded_attempts':len(superseded),'total_recorded_attempts':len(all_rows),
              'not_run_with_reason':blocked,'pending':[c['id'] for c in absent]}
    baselines=[]
    for key in sorted({c['baseline_key'] for c in m['cases']}):
        samples=[r for r in rows if r['case'].get('baseline') and
                 r['case']['baseline_key']==key and r['status']=='complete']
        if not samples:continue
        walls=[r['wall_seconds'] for r in samples]
        median=statistics.median(walls)
        baselines.append({'key':key,'ids':[r['case']['id'] for r in samples],
            'samples':len(samples),'wall_seconds':walls,'median_wall_seconds':median,
            'range_seconds':[min(walls),max(walls)],
            'spread_percent':100*(max(walls)-min(walls))/median if len(walls)>1 else None,
            'peak_vram_gib':[r['max_vram_gib'] for r in samples],
            'median_peak_vram_gib':statistics.median(r['max_vram_gib'] for r in samples)})
    telemetry=[{'case':r['case']['id'],**{k:r.get('telemetry',{}).get(k)
               for k in ['samples','max_gap_seconds','gaps_over_100ms','error']},
               'gpu_released':r.get('gpu_released')} for r in rows]
    result={'manifest_identity':m['identity'],'coverage':coverage,'records':rows,
            'gpu_name':environment['gpu_name'],'gpu_uuid':environment['gpu_uuid'],
            'environment_file':'environment.json',
            'extension_identity':m.get('extension_identity'),
            'final_identity':m.get('final_identity'),
            'sampling_identity':m.get('sampling_identity'),'interruptions':interruptions,
            'baselines':baselines,'telemetry_quality':telemetry,
            'embedded_audio_comparisons':audio_comparisons,
            'superseded_attempts':superseded,'retry_decisions':retry_decisions,
            'repeat_comparisons':repeats,'harness_corrections':corrections,
            'visual_observations':observations,
            'measurement_notes':['Whole CLI monotonic wall time; fresh process, filesystem cache uncontrolled',
                'VRAM is sampled whole-device high water, including idle context/driver use',
                'Per-process peak and initial idle usage are separate NVML counters; neither is subtracted from the primary whole-device peak',
                'Reference/no-reference timing includes different Ref2VA and T2VA/FL2VA routes',
                'Single candidate runs and few denoising steps do not qualify high-step final-render quality',
                'The user replaced the unfinished ten-step matrix with a one-hour sample capped at two steps; the selected matched 362-frame pairs use one step',
                'Sample pairs have one matched control each, with no repeat-based estimate of timing variability; steps and geometry are never mixed in comparisons',
                'O03/O91 add binding clauses; core comparisons use identical literal prompts']}
    save(ROOT/'results.json',result)
    save(ROOT/'bugs.json',{'source_identity':m['identity'],'product_fixes':'on hold',
        'bugs':list(bugs.values()),'harness_corrections':corrections,'visual_observations':observations,
        'interruptions':interruptions,
        'superseded_attempts':superseded,'retry_decisions':retry_decisions})
    bugtext=['# CUDA multi-reference findings','', 'Product fixes remain on hold.','']
    for b in bugs.values():
        bugtext += [f'## {b["id"]}: {b["title"]}',f'Category: {b["category"]}; severity: {b["severity"]}.',
                    f'Cases: {", ".join(b["cases"])}. Fix status: {b["fix_status"]}.','',
                    f'Expected: {b["expected"]}.','']
        for x in b.get('evidence',[]):
            bugtext += [f'- {x["case"]}: {x["observed"]}',
                        f'  [Command]({x["request"]}) · [Log]({x["stderr"]}) · [Telemetry]({x["telemetry"]})']
        bugtext.append('')
    if not bugs: bugtext+=['No confirmed runtime findings in the measured cases so far.','']
    if interruptions:
        bugtext+=['## User-requested interruption','']
        for name,detail in interruptions.items():
            bugtext+=[f'- {name}: {detail["reason"]}. This is not a renderer failure. '
                      f'[Original metrics]({detail["raw_metrics"]}) · [Budget decision]({detail["decision"]}).']
        bugtext.append('')
    if corrections:
        bugtext+=['## Test-tool and environment corrections','']
        for c in corrections:
            bugtext+=[f'- {c["id"]}: {c["issue"]}. {c["status"]}. '
                      f'[Retained evidence]({c["excluded_record"]}).']
    if observations:
        bugtext+=['','## Visual observations (unconfirmed)','']
        for o in observations:
            bugtext+=[f'- {o["case"]}: {o["observation"]} {o["classification"]}. '
                      f'[Evidence]({o["evidence"]}).']
    if superseded:
        bugtext+=['','## Retained attempts superseded by explicit retries','']
        for r in superseded:
            bugtext+=[f'- {r["case"]["id"]}, {r["artifact_directory"]}: '
                      f'{r["status"]}; {r.get("termination_reason")}; '
                      f'{number(r.get("wall_seconds"))} s / {number(r.get("max_vram_gib"))} GiB. '
                      f'[Metrics]({r["artifact_directory"]}/metrics.json) · '
                      f'[Log]({r["artifact_directory"]}/stderr.log).']
    (ROOT/'bugs.md').write_text('\n'.join(bugtext)+'\n')
    fields=['case','phase','resolution','frames','steps','images','videos','image_size','status',
            'wall_seconds','baseline_ids','baseline_wall_seconds','wall_delta_seconds','wall_over_baseline',
            'wall_overhead_percent','max_vram_gib','process_peak_vram_gib','idle_vram_gib','baseline_vram_gib','vram_delta_gib',
            'completed_steps','output','bug_ids']
    table=[]
    for c in m['cases']:
        r=by_id.get(c['id'],{})
        comparison=r.get('comparison',{})
        table.append({'case':c['id'],'phase':c['phase'],'resolution':f'{c["width"]}x{c["height"]}',
            'frames':c['frames'],'steps':c['steps'],'images':sum(m['assets'][k]['kind']=='image' for k in c['references']),
            'videos':sum(m['assets'][k]['kind']=='video' for k in c['references']),
            'image_size':c['image_size'],'status':r.get('status','not run: '+blocked[c['id']] if c['id'] in blocked else 'pending'),
            **{k:r.get(k) for k in ['wall_seconds','max_vram_gib','process_peak_vram_gib','idle_vram_gib','completed_steps','output']},
            **{k:comparison.get(k) for k in ['baseline_wall_seconds','wall_delta_seconds','wall_over_baseline',
                'wall_overhead_percent','baseline_vram_gib','vram_delta_gib']},
            'baseline_ids':';'.join(comparison.get('baseline_ids',[])), 'bug_ids':';'.join(r.get('bug_ids',[]))})
    with (ROOT/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(table)
    text=['# CUDA multi-reference results','',
          f'{coverage["complete"]} completed; {coverage["failed_or_blocked_attempts"]} measured failures/blocks; '
          f'{coverage["interrupted_by_user"]} interrupted by user; '
          f'{len(blocked)} unrun with reasons; {len(absent)} pending, of {len(scheduled)} scheduled attempts.','',
          '[Playback index](review.html) · [CSV](results.csv) · [Full records](results.json) · [Findings](bugs.md)','',
          f'{environment["gpu_name"]}, {environment["vram_total"]/(1<<30):.2f} GiB device memory; '
          f'driver {environment["driver"]}. Resident original BF16 weights, default dense attention, '
          'full VAE; retained main cases use ten steps, smoke uses two, and the one-hour sample uses matched one-step pairs.','',
          '[Environment](environment.json) · [Final configuration](final-main.json) · '
          '[Source identity](source.json) · [Model identities](model-identity.json)','',
          '| Case | Status | Wall s | Baseline s | Overhead % | Peak VRAM GiB | VRAM Δ GiB | Output |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for t in table:
        link=f'[MP4]({t["output"]})' if t['output'] else '—'
        text.append(f'| {t["case"]} | {t["status"]} | {number(t["wall_seconds"])} | '
                    f'{number(t["baseline_wall_seconds"])} | {number(t["wall_overhead_percent"])} | '
                    f'{number(t["max_vram_gib"])} | {number(t["vram_delta_gib"])} | {link} |')
    text+=['','## Measurement limits','']+['- '+s for s in result['measurement_notes']]
    if superseded:
        text+=['','## Retained earlier attempts','',
               'These attempts are counted separately and excluded from successful-render comparisons.','',
               '| Case / attempt | Status | Wall s | Peak VRAM GiB | Reason |',
               '| --- | --- | ---: | ---: | --- |']
        for r in superseded:
            text.append(f'| [{r["artifact_directory"]}]({r["artifact_directory"]}/metrics.json) | '
                        f'{r["status"]} | {number(r.get("wall_seconds"))} | '
                        f'{number(r.get("max_vram_gib"))} | {r.get("termination_reason")} |')
    text+=['','## Baseline samples','',
           '| Group | Wall samples s | Median s | Spread % | Peak VRAM samples GiB |',
           '| --- | --- | ---: | ---: | --- |']
    for b in baselines:
        text.append(f'| {b["key"]} | '+', '.join(number(v) for v in b['wall_seconds'])+
                    f' | {number(b["median_wall_seconds"])} | {number(b["spread_percent"])} | '+
                    ', '.join(number(v) for v in b['peak_vram_gib'])+' |')
    gaps=[t for t in telemetry if t['gaps_over_100ms']]
    text+=['','## Telemetry coverage','',
           f'{len(gaps)} measured cases have sampling gaps above the 100 ms target maximum. '
           'Their peaks remain sampled lower bounds; see the per-case gap counts and maximum '
           'intervals in results.json. Missing NVML measurements invalidate an attempt.']
    checked=[r for r in rows if r.get('black_frame_check')]
    text+=['','## Blank-output screening','',
           f'{len(checked)} validated outputs checked with FFmpeg blackdetect. '
           f'{sum(r["black_frame_check"]["warning"] for r in checked)} have a nearly-black span '
           'of at least 0.5 seconds (98% of pixels below 2% luma). '
           'This heuristic flags possible visual issues; it does not change render completion status. '
           'Decoded-frame hashes separately flag nearly frozen output.']
    text+=['','## Same-input repeat checks','', '```json',json.dumps(repeats,indent=2),'```']
    text+=['','## Embedded audio versus the same silent video','',
           'These comparisons include the audio encoder and audio conditioning costs. '
           'The input video stream is unchanged; the A1 fixture adds a synthetic tone.','',
           '```json',json.dumps(audio_comparisons,indent=2),'```']
    unstable=sorted({r['case']['baseline_key'] for r in rows if r.get('comparison',{}).get('baseline_unstable')})
    text+=['',f'Baseline groups with >10% observed wall-time spread: {", ".join(unstable) or "none yet"}.',
           'Each comparison requires its own resolution, frame count and evaluation count. '
           'Incomplete baseline groups remain provisional.','']
    (ROOT/'report.md').write_text('\n'.join(text)+'\n')
    style='body{max-width:1500px;margin:30px auto;padding:0 20px;font:16px/1.5 system-ui;background:#f3f4f6;color:#171923}section{padding:20px;background:white;margin:24px 0;border-radius:10px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}video,img{max-width:100%;height:auto}.refs{display:flex;gap:12px;flex-wrap:wrap}.refs figure{width:150px;margin:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.bad{border-left:6px solid #b42318}a{color:#125194}button{padding:8px 14px;margin:5px}'
    script="""<script>
    function pair(id,play){let v=[...document.getElementById(id).querySelectorAll('.grid video')];v.forEach(x=>{x.currentTime=0;if(play)x.play().catch(()=>{});else x.pause()})}
    document.querySelectorAll('section').forEach(s=>{let v=[...s.querySelectorAll('.grid video')];if(v.length!==2)return;
      let a=v[0],b=v[1];b.muted=true;
      a.addEventListener('timeupdate',()=>{if(b.readyState&&Math.abs(a.currentTime-b.currentTime)>.12)b.currentTime=a.currentTime});
      a.addEventListener('play',()=>{b.currentTime=a.currentTime;b.play().catch(()=>{})});
      a.addEventListener('pause',()=>b.pause());a.addEventListener('seeking',()=>{if(b.readyState)b.currentTime=a.currentTime});
    });</script>"""
    def page(title,body):
        return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+e(title)+'</title><style>'+style+'</style><body><h1>'+e(title)+'</h1>'+body+script+'</body></html>'
    nav='<p><a href="review.html">Index</a> · <a href="report.md">Report</a> · <a href="results.csv">CSV</a> · <a href="bugs.md">Findings (fixes deferred)</a></p>'
    for width,height in [(640,480),(1344,768)]:
        body=nav+'<p>Dense BF16, full VAE. Retained main cases: ten steps. Smoke: two steps. One-hour sample: matched one-step pairs, within the two-step cap. VRAM is sampled whole-device usage. Inputs and outputs may contain nudity.</p>'
        for c in m['cases']:
            if c['width']!=width: continue
            r=by_id.get(c['id'])
            body+=f'<section id="{e(c["id"])}" class="{"bad" if not r or r["status"]!="complete" else ""}"><h2>{e(c["id"])}</h2>'
            body+=f'<p>{c["frames"]} frames, {c["steps"]} steps; sizing {c["image_size"]}; {e(c["phase"])}.</p>'
            if not r:
                body+=f'<p>{e(blocked.get(c["id"],"Pending"))}</p></section>'
                continue
            v=r['comparison']
            body+=f'<p>Status: {e(r["status"])}. Wall {number(r.get("wall_seconds")," s")}; peak VRAM {number(r.get("max_vram_gib")," GiB")}; overhead {number(v.get("wall_overhead_percent"),"%")}. '
            body+=f'Baseline {number(v.get("baseline_wall_seconds")," s")}, samples {v.get("baseline_samples",0)}.</p>'
            body+=f'<p>Process peak {number(r.get("process_peak_vram_gib")," GiB")}; idle device usage {number(r.get("idle_vram_gib")," GiB")}.</p>'
            if r.get('interruption'):body+='<p>'+e(r['interruption']['reason'])+'; partial timing retained, no completed output.</p>'
            if v.get('baseline_unstable'): body+='<p>Baseline timing spread exceeds 10%; comparison is provisional.</p>'
            if c['bound_prompt']: body+='<p>This ordering case adds explicit prompt bindings; baseline omits that clause.</p>'
            body+='<p>'+e(c['prompt'])+'</p>'
            body+='<div class="refs">'
            counters={'image':0,'video':0}
            for k in c['references']:
                a=m['assets'][k];counters[a['kind']]+=1
                label=('Picture' if a['kind']=='image' else 'Video')+' '+str(counters[a['kind']])
                if a['kind']=='image':
                    size=a['canvases'][f'{width}x{height}-{c["image_size"]}']
                    detail=f'{a["width"]}×{a["height"]} → {size[0]}×{size[1]}; '
                    detail+=f'latent 1×{size[1]//16}×{size[0]//16}, {(size[0]//32)*(size[1]//32)} conditioning rows (host-computed)'
                    media=f'<img loading="lazy" src="{a["thumbnail"]}" alt="{e(label)}">'
                else:
                    detail=f'{a["frames"]} input frames, {a["duration"]:.3f} s; VAE {a["vae_frames"]}, latent T {a["latent_t"]}'
                    media=f'<video controls preload="none" poster="{a["thumbnail"]}" src="fixtures/{Path(a["path"]).name}"></video>'
                body+=f'<figure>{media}<figcaption>{e(label)}: {e(k)}<br>{e(detail)}</figcaption></figure>'
            body+='</div>'
            if r.get('output'):
                bases=[by_id[k] for k in v.get('baseline_ids',[]) if by_id[k].get('output')]
                baseline=min(bases,key=lambda x:abs(x['wall_seconds']-v['baseline_wall_seconds'])) if bases else None
                body+='<div class="grid"><figure><video controls preload="none" src="'+e(r['output'])+'"></video><figcaption>Candidate ('+e(r['status'])+')</figcaption></figure>'
                if baseline:
                    body+='<figure><video controls preload="none" src="'+e(baseline['output'])+'"></video><figcaption>No references: '+e(baseline['case']['id'])+'</figcaption></figure>'
                body+='</div><button onclick="pair(\''+e(c['id'])+'\',true)">Play pair from start</button><button onclick="pair(\''+e(c['id'])+'\',false)">Reset pair</button>'
                if r.get('contact_sheet'): body+=f'<p><img loading="lazy" src="{e(r["contact_sheet"])}" alt="Six output timestamps"></p>'
            if r['bug_ids']: body+='<p>Findings: '+e(', '.join(r['bug_ids']))+'</p>'
            if r.get('black_frame_check'):
                blank=r['black_frame_check']
                body+=f'<p>Nearly-black spans ≥0.5 s: {len(blank["intervals"])}. '
                body+=f'<a href="{e(blank["log"])}">Blank-frame screening log</a> (visual heuristic).</p>'
            for o in r['visual_observations']:
                body+='<p>Visual observation: '+e(o['observation'])+' '+e(o['classification'])+'</p>'
            body+=f'<p><a href="{r["artifact_directory"]}/request.json">Exact command</a> · <a href="{r["artifact_directory"]}/stderr.log">Log</a> · <a href="{r["artifact_directory"]}/metrics.json">Metrics</a></p></section>'
        (ROOT/f'review-{width}x{height}.html').write_text(page(f'CUDA references: {width}×{height}',body))
    body=nav+f'<p>{coverage["complete"]} complete, {coverage["failed_or_blocked_attempts"]} measured failures/blocks, {len(blocked)} not run with reasons, {len(absent)} pending.</p>'
    if interruptions:body+=f'<p>{len(interruptions)} ten-step run interrupted at the user’s request; raw records are retained.</p>'
    if m.get('sampling_identity'):body+='<p>The remaining matrix was replaced by a one-hour sample capped at two steps. <a href="sample-one-hour.json">Sample selection and deadline</a>. Selected one-step 362-frame pairs use matching no-reference controls where both runs completed; missing controls leave comparisons unavailable.</p>'
    body+=f'<p>{e(environment["gpu_name"])}; {environment["vram_total"]/(1<<30):.2f} GiB; driver {e(environment["driver"])}. '
    body+='<a href="environment.json">Environment</a> · <a href="final-main.json">Final configuration</a> · <a href="source.json">Source identity</a>.</p>'
    if superseded:
        body+='<h2>Retained earlier attempts</h2><ul>'
        for r in superseded:
            body+=f'<li>{e(r["artifact_directory"])}: {e(r.get("termination_reason"))}, '
            body+=f'{number(r.get("wall_seconds"))} s / {number(r.get("max_vram_gib"))} GiB. '
            body+=f'<a href="{e(r["artifact_directory"])}/metrics.json">Metrics</a> · '
            body+=f'<a href="{e(r["artifact_directory"])}/stderr.log">Log</a></li>'
        body+='</ul>'
    body+='<p><a href="review-640x480.html">640×480 playback</a> · <a href="review-1344x768.html">1344×768 playback</a></p>'
    body+='<p>Default released-v1 references, original BF16 checkpoints, full VAE. Up to nine images and three videos / six seconds total. Product fixes remain on hold. The two synthetic zoom fixtures have limited real-motion evidence.</p>'
    (ROOT/'review.html').write_text(page('CUDA multi-reference test campaign',body))
    print('REPORT',json.dumps(coverage),flush=True)


if __name__=='__main__': report()
