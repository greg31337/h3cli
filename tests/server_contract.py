#!/usr/bin/env python3
"""CPU-only request/registry tests; no model loading or network service."""
import json, os, re, shlex, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
BINARY=os.environ.get('H3_SERVER_CONTRACT_BINARY','bin/server_contract')
checks=0
def normalize(body,good=True,contains=None,mode=None):
 global checks
 p=subprocess.run([BINARY]+([mode] if mode else []),input=json.dumps(body,ensure_ascii=False),text=True,capture_output=True)
 checks+=1
 if good:
  assert p.returncode==0,(body,p.stderr)
  r=json.loads(p.stdout);r['map']={v['name']:v['value'] for v in r['options']};return r
 assert p.returncode==2,(body,p.returncode,p.stdout,p.stderr)
 if contains: assert contains in p.stderr,(contains,p.stderr,body)
def invalid_string(flags):normalize({'h3cli':flags},False)
registry=json.loads(subprocess.check_output([BINARY,'registry']))
assert len(registry)==len({v['name'] for v in registry})
assert next(v for v in registry if v['name']=='seed')['maximum']==2**64-1
source=(ROOT/'src/h3cli.c').read_text()
ids=re.findall(r'H3_OPTION\("([^"]+)", ([^,]+),', (ROOT/'src/cli/options.def').read_text())
for d in registry:
 name=d['name'];assert any(n==name for n,_ in ids)
 enum=next(i for n,i in ids if n==name)
 if name not in ('models-path','download-models','list-models','offline'):assert 'case '+enum+':' in source,(name,enum)
 values=[]
 if d['arity']:
  value=d['choices'].split('|')[0] if d['choices'] else ('path with unicode 猫' if d['type']=='string' else str(max(d['minimum'],1)))
  if d['type'] in ('integer','uint64'):value=str(int(float(value)))
  values=[value]+(['sound track.wav'] if d['arity']==2 else [])
 flags=' '.join(['--'+name]+[shlex.quote(v) for v in values])
 if d['operations']=='process':
  normalize({'h3cli':flags},False);continue
 r=normalize({'h3cli':flags},mode='parse-only');assert r['map'][name]==(values[0] if values else True)
 if d.get('short'):
  rr=normalize({'h3cli':' '.join(['-'+d['short']]+[shlex.quote(v) for v in values])},mode='parse-only');assert rr['options']==r['options']
 if values:
  rr=normalize({'h3cli':'--'+name+'='+shlex.quote(values[0])+(' '+shlex.quote(values[1]) if len(values)>1 else '')},mode='parse-only');assert rr['options']==r['options']
 if d['type'] in ('integer','uint64','number'):
  for val in ('nan','inf','-1','1e999','18446744073709551616'):
   if val=='-1' and d['minimum']<=-1:continue
   if d['type']=='number' and val=='18446744073709551616' and d['maximum']>=2**64:continue
   normalize({'h3cli':'--'+name+' '+val},False)
 if d['type']=='enum':normalize({'h3cli':'--'+name+' invalid'},False)
for flags in ['--server','--server-port 8','--server-worker','--attention dense','--metal--attention dense','--cuda-reference','--legacy-ref2va-video-pipeline','h3cli --help','echo hi','--prompt "unfinished','--prompt hi\\','--ref-video-audio only','--help=true','--seed 1.5','--width 1e3','--width +2','--seed -0','--help operands','--prompt x'*7000]:invalid_string(flags)
for value in (None,[],{},1,False):normalize({'h3cli':value},False)
for name in ('h3','options','references','loras','operation','environment'):
 normalize({'h3cli':'--help',name:{}},False)
for flag in ["-p '猫 🌲'",'-p "a b"',"--prompt=''",'-p "$(touch /tmp/SHOULD_NOT_EXIST) `id` $HOME *.jpg"']:
 normalize({'h3cli':flag},mode='parse-only')
assert not Path('/tmp/SHOULD_NOT_EXIST').exists()
fixtures=json.loads((ROOT/'tests/fixtures/server/sglang.json').read_text())
r=normalize(fixtures['canonical']);assert (r['map']['width'],r['map']['height'],r['map']['frames'],r['map']['steps'])==('256','256','107','2')
r=normalize(fixtures['conflicting']);assert r['seeds']==[42] and r['map']['width']=='256' and r['map']['steps']=='2'
for quality in ('lossless','extra-high','high'):
 for order in ('--steps 2 --quality '+quality,'--quality '+quality+' --steps 2'):
  r=normalize({**fixtures['canonical'],'quality':'high','h3cli':order});assert r['map']['steps']=='2' and r['map']['quality']==quality
for quality in ('preview','fast-preview'):
 normalize({**fixtures['canonical'],'quality':quality},False,'quality')
 r=normalize({**fixtures['canonical'],'h3cli':'--quality '+quality});assert 'steps' not in r['map']
 r=normalize({**fixtures['canonical'],'h3cli':'--quality '+quality+' --steps 6 --no-preview-vae'});assert r['map']['steps']=='6'
for quality in ('default','maximum','high','medium','low'):
 r=normalize({**fixtures['canonical'],'output_quality':quality});assert r['map']['output-quality']==quality
normalize({**fixtures['canonical'],'output_quality':'invalid'},False,'output-quality')
assert 'output-quality' not in normalize(fixtures['canonical'])['map']
for flags in ('--output-quality low','--ffmpeg-crf 18','--lossless-video'):
 r=normalize({**fixtures['canonical'],'output_quality':'invalid','h3cli':flags})
 assert ('output-quality' in r['map'])==flags.startswith('--output-quality')
for flags in ('--ffmpeg-crf 18 --output-quality low','--output-quality low --ffmpeg-crf 18'):
 r=normalize({**fixtures['canonical'],'output_quality':'maximum','h3cli':flags})
 assert r['map']['ffmpeg-crf']=='18' and r['map']['output-quality']=='low'
for wrapper in ('extra_body','extra_json','extra_params'):
 r=normalize({wrapper:fixtures['conflicting']});assert r['map']==normalize(fixtures['conflicting'])['map']
 normalize({'h3cli':'--help',wrapper:{'h3cli':'--info'}},False,'conflicting')
base={'prompt':'x','h3cli':'--width 256 --height 256 --frames 22 --steps 2'}
assert normalize({**base,'seed':2**64-1})['seeds']==[2**64-1]
assert normalize({**base,'seed':[2,4,8],'n':3})['seeds']==[2,4,8]
assert normalize({**base,'seed':[2,4,8],'n':3,'h3cli':base['h3cli']+' --seed 42'})['seeds']==[42,43,44]
for body in ({'seed':2**64},{'seed':-1},{'seed':1.0},{'n':2,'seed':2**64-1},{'seed':[1,2]},{'n':2,'num_outputs_per_prompt':3}):normalize({**base,**body},False)
condition=lambda t,u,r='reference',**kw:dict(type=t,uri=u,role=r,**kw)
refs=[condition('image','missing.jpg'),condition('video','http://127.0.0.1/forbidden'),condition('audio','/etc/passwd')]
r=normalize({**base,'task':'ref2va','conditions':refs,'h3cli':base['h3cli']+' --ref-image new.jpg --ref-video-audio vid.mp4 snd.wav --ref-audio a.wav'})
assert [v['value'] for v in r['options'] if v['name'].startswith('ref-')]==['new.jpg','vid.mp4','a.wav']
assert next(v for v in r['options'] if v['name']=='ref-video-audio')['second']=='snd.wav'
anchors=[condition('image','old-first','keyframe',frame_index=0),condition('image','keep-last','keyframe',frame_index=-1)]
r=normalize({**base,'task':'fl2va','conditions':anchors,'h3cli':base['h3cli']+' --first-frame new-first'});assert r['map']['first-frame']=='new-first' and r['map']['last-frame']=='keep-last'
r=normalize({**base,'conditions':anchors,'h3cli':base['h3cli']+' --ref-image reference'});assert 'first-frame' not in r['map'] and 'last-frame' not in r['map']
r=normalize({**base,'conditions':refs,'h3cli':base['h3cli']+' --first-frame first'});assert not any(v['name']=='ref-image' for v in r['options'])
r=normalize({**base,'conditions':refs,'h3cli':base['h3cli']+' --ref-image-size max'});assert r['map']['ref-image']=='missing.jpg'
for flag,op in [('decode-av-state','decode_av'),('resume-sampler-state','resume'),('decode-still-latent','decode_still'),('upscale-state','upscale'),('inspect-upscale-state','inspect_upscale')]:
 r=normalize({**fixtures['canonical'],'conditions':refs,'h3cli':'--'+flag+' saved'});assert r['operation']==op and 'prompt' not in r['map'] and 'steps' not in r['map'] and not any(v['name'].startswith('ref-') for v in r['options'])
for target,dims in [({'short_edge':256,'aspect_ratio':'16:9','duration_seconds':4},('448','256','107')),({'short_edge':2048,'aspect_ratio':'16:9','duration_seconds':15},('1344','768','362')),({'short_edge':272,'aspect_ratio':'1:1','duration_seconds':4},('256','256','107')),({'short_edge':304,'aspect_ratio':'1:1','duration_seconds':4},('320','320','107'))]:
 r=normalize({**fixtures['canonical'],'target':target});assert tuple(r['map'][k] for k in ('width','height','frames'))==dims,(target,r)
for field,value in [('fps',24),('num_frames',22),('enable_cache_dit',True),('enable_teacache',True),('guidance_scale',1),('negative_prompt',''),('flow_shift',1),('audio_flow_shift',2),('video_path','a'),('task_type','x')]:normalize({**base,field:value},False,field)
for field in ('enable_cache_dit','enable_teacache','enable_upscaling','enhance_prompt'):normalize({**base,field:False})
for raw in ('{"h3cli":"--help","h3cli":"--info"}','{"h3cli":"--prompt \\u0000"}','{"h3cli":"--help","seed":NaN}'):
 p=subprocess.run([BINARY],input=raw,text=True,capture_output=True);checks+=1;assert p.returncode==2
r=normalize({**base,'conditions':[condition('video','with-soundtrack.mp4')]});assert r['map']['ref-video']=='with-soundtrack.mp4'
normalize({**fixtures['canonical'],'target':{'short_edge':256,'aspect_ratio':'999999999999999999999:1','duration_seconds':4}},False)
normalize({'h3cli':'--sol-dense-sigma -1'},mode='parse-only')
report={'checks':checks,'options':len(registry),'option_inventory':registry,'cuda_execution':False,'gpu_execution':False}
out=ROOT/'outputs/server-validation/contract';out.mkdir(parents=True,exist_ok=True);(out/('sanitize.json' if 'sanitize' in BINARY else 'coverage.json')).write_text(json.dumps(report,indent=2)+'\n')
print(f'{checks} server contract checks passed; {len(registry)} options mapped')
