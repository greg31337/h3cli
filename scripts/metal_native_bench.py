#!/usr/bin/env python3
"""Reproducible M0/M1 save-once/load-many workloads; never exceed six evaluations."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = {'B1': 1, 'B2': 2, 'B5': 5, 'B6': 6}
PROMPT = 'A person in a blue jacket walks through a sunny park. Birds sing softly.'

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            digest.update(data)
    return digest.hexdigest()

def command_output(command, env=None):
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    return {'command': command, 'returncode': result.returncode,
            'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}

def manifest(binary, model, env, original_binary=None):
    sources = [*ROOT.glob('src/**/*.c'), *ROOT.glob('src/**/*.h'), *ROOT.glob('src/**/*.m'),
               *ROOT.glob('src/**/*.metal'), *ROOT.glob('src/**/*host.inc'), *ROOT.glob('third_party/mlx-attention/**/*'),
               ROOT/'Makefile', ROOT/'scripts/embed_metal_attention.py', Path(__file__)]
    sources = [p for p in sources if p.is_file()]
    binary_sha=sha(binary)
    provenance=None
    if original_binary:
        companion=original_binary.parent/'build-provenance.json'
        if companion.is_file():
            candidate=json.loads(companion.read_text())
            if candidate.get('binary_sha256')!=binary_sha:raise ValueError('Stale binary build provenance')
            snapshot=Path(candidate['source_snapshot'])
            for name,digest in candidate['source_sha256'].items():
                if sha(snapshot/name)!=digest:raise ValueError('Changed binary source snapshot: '+name)
            provenance=candidate
    return {'binary_sha256': binary_sha, 'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(sources)},
            'source_hash_scope': 'Working tree at launch; binary_build_provenance, when present, identifies the frozen executable sources.',
            'binary_build_provenance':provenance,
            'git_commit': command_output(['git', 'rev-parse', 'HEAD']),
            'git_status': command_output(['git', 'status', '--short']),
            'platform': platform.platform(), 'python': sys.version,
            'device': command_output([str(binary), '-d', str(model), '--info'], env),
            'os': command_output(['sw_vers']), 'compiler': command_output(['clang', '--version']),
            'xcode': command_output(['xcodebuild', '-version'], env),
            'environment': {k: v for k, v in env.items() if k.startswith('H3_') or k == 'DEVELOPER_DIR'}}

def statistics_for(steps):
    values = [r['wall_seconds'] for r in steps]
    steady = values[1:]
    return {'step_seconds': values, 'first_step_seconds': values[0] if values else None,
            'steady_steps': list(range(2, len(values)+1)),
            'steady_median': statistics.median(steady) if steady else None,
            'steady_mean': statistics.mean(steady) if steady else None,
            'steady_minimum': min(steady) if steady else None,
            'steady_maximum': max(steady) if steady else None,
            'summed_denoise_seconds': sum(values)}

def conditioning_identity(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        header=stream.read(128)
        if len(header)!=128 or header[:8]!=b'H3COND\0\0':raise ValueError('Invalid cache header')
        count=struct.unpack_from('<I',header,24)[0]
        if not 1<=count<=80:raise ValueError('Invalid cache record count')
        table=stream.read(count*96)
        if len(table)!=count*96:raise ValueError('Truncated cache table')
        for offset in range(0,len(table),96):
            if struct.unpack_from('<I',table,offset)[0] == 1:
                start,length=struct.unpack_from('<QQ',table,offset+40)
                if not 0<length<=1<<20:raise ValueError('Invalid cache identity length')
                stream.seek(start);data=stream.read(length)
                if len(data)!=length:raise ValueError('Truncated cache identity')
                return data.decode('utf-8')
    raise ValueError('Cache has no identity')

def sol_policy_check(record):
    """Match every block's effective routing to the absolute step and both sigmas."""
    decisions=record['sol_policy'];routes=record['sol'];options=record['metal_options']
    expected={(s,b) for s in range(1,record['evaluations']+1) for b in range(record['blocks'])}
    if len(decisions)!=len(expected) or {(r['step'],r['block']) for r in decisions}!=expected:return False
    routed=set()
    for r in decisions:
        sigmas=[r['video_sigma'],r['audio_sigma']]
        if any(not math.isfinite(s) or not 0<=s<=1 for s in sigmas):return False
        if options['sol-min-exact']>=1:reason='all-exact'
        elif r['step']-1<options['sol-dense-steps']:reason='early-evaluation'
        elif options['sol-dense-sigma']>=0 and max(sigmas)>=options['sol-dense-sigma']:reason='high-noise'
        elif r['block']<options['sol-dense-layers']:reason='early-layer'
        else:reason='routed';routed.add((r['step'],r['block']))
        if r['decision']!=reason:return False
    if len(routes)!=len(routed) or {(r['step'],r['block']) for r in routes}!=routed:return False
    return all(r['exact']>0 and r['approximate']>=0 and 0<=r['protected']<=r['exact'] and
               0<=r['local']<=r['exact']-r['protected'] and r['protected_rows']>0 for r in routes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, default=ROOT/'bin/h3cli')
    p.add_argument('--model', type=Path, default=ROOT/'models/MiniMax-H3')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--conditioning', type=Path, required=True)
    p.add_argument('--save-conditioning', action='store_true', help='Create cache once while measuring the uncached baseline')
    p.add_argument('--schedule-cache', action='store_true', help='Save/require optional exact-schedule AdaLN records')
    p.add_argument('--case', choices=CASES, default='B1')
    p.add_argument('--backend', choices=['mpsgraph', 'metal'], default='mpsgraph')
    p.add_argument('--metal-attention', dest='attention', choices=['dense','sol'], default='dense')
    p.add_argument('--metal-attention-kernel', choices=['steel','steel-routed'], default='steel')
    p.add_argument('--metal-attention-dtype', choices=['bf16','fp16'], default='bf16')
    p.add_argument('--metal-attention-layout', choices=['adapter','fused'], default='adapter')
    p.add_argument('--metal-tier', choices=['diagnostic','reference','preview'], default='diagnostic')
    p.add_argument('--metal-ane', choices=['off','serial','static','dynamic'], default='off')
    p.add_argument('--metal-ane-rows', type=int, default=4096)
    p.add_argument('--metal-ane-chunk', type=int, default=512)
    p.add_argument('--metal-weight-format', choices=['bf16','q8'], default='bf16')
    p.add_argument('--metal-q8-kernel', choices=['mpsgraph','simdgroup'], default='mpsgraph')
    p.add_argument('--sol-q-block', type=int, choices=[32,64], default=32)
    p.add_argument('--sol-kv-block', type=int, choices=[32,64,128], default=64)
    p.add_argument('--sol-tau', type=float, default=1.)
    p.add_argument('--sol-dense-layers', type=int, default=1)
    p.add_argument('--sol-dense-steps', type=int, default=0)
    p.add_argument('--sol-dense-sigma', type=float, default=-1.)
    p.add_argument('--sol-local-radius', type=int, default=1)
    p.add_argument('--sol-min-exact', type=float, default=.1)
    p.add_argument('--blocks', type=int, choices=[1, 5, 50], default=50)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--frames', type=int, default=243)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prompt', default=PROMPT)
    p.add_argument('--reference', type=Path, action='append', default=[], help='Repeat for a reference-heavy image workload')
    p.add_argument('--ref-video', type=Path)
    p.add_argument('--ref-audio', type=Path)
    p.add_argument('--first-frame', type=Path)
    p.add_argument('--last-frame', type=Path)
    p.add_argument('--continue-from', type=Path)
    p.add_argument('--continue-context', type=int, choices=[39,90,141,192], default=39)
    p.add_argument('--full-vae', action='store_true', help='Explicitly opt out of development preview VAE')
    p.add_argument('--preview-model', type=Path, default=ROOT/'models/taeh3.safetensors')
    p.add_argument('--components', action='store_true', help='Fence component intervals; diagnostic B1 only')
    p.add_argument('--regions', action='store_true', help='Fence complete attention/block regions without component fences; diagnostic B1 only')
    p.add_argument('--capture-boundaries', action='store_true', help='Save first-block projection and block outputs; diagnostic B1 only')
    p.add_argument('--capture-steps', action='store_true', help='Save CPU Euler per-step velocities and latents; diagnostic timing only')
    p.add_argument('--teacher-from', type=Path, help='Diagnostic only: replace each later input with this reference run’s saved latent')
    p.add_argument('--capture-ranges', action='store_true', help='GPU range scans and sampled score/softmax probes; diagnostic timing only')
    p.add_argument('--trace', action='store_true', help='Capture a B1 Metal System Trace using scoped Xcode')
    p.add_argument('--trace-window', type=int, default=20, help='Bounded denoising trace window in seconds (1..60)')
    p.add_argument('--capture-qkv', action='store_true', help='Retain block-zero real QKV for the independent attention oracle')
    p.add_argument('--timeout', type=int, default=7200)
    a = p.parse_args()
    extra_inputs={k:getattr(a,k) for k in ('ref_video','ref_audio','first_frame','last_frame') if getattr(a,k)}
    teacher = None
    teacher_hashes = {}
    if a.teacher_from:
        a.teacher_from = a.teacher_from.resolve()
        teacher = json.loads((a.teacher_from/'record.json').read_text())
        expected = {'geometry':[a.width,a.height,a.frames], 'seed':a.seed, 'prompt':a.prompt,
                    'case':a.case, 'blocks':a.blocks, 'backend':'mpsgraph', 'sampler':'cpu-euler',
                    'capture_steps':True, 'evaluation_contract_pass':True}
        if any(teacher.get(k)!=v for k,v in expected.items()) or not a.capture_steps or a.reference or a.continue_from or extra_inputs:
            p.error('teacher input requires matching captured T2VA MPSGraph run, geometry, prompt, seed, schedule and --capture-steps')
        if teacher.get('teacher_from') or sha(a.teacher_from/'result.h3av') != teacher['av_sha256']:
            p.error('teacher input is stale or itself teacher-forced')
        for step in range(1, CASES[a.case]+1):
            for modality in ('video','audio'):
                for tensor in ('velocity','latent'):
                    name=f'step-{step:03d}-{modality}-{tensor}.f32'
                    teacher_hashes[name]=sha(a.teacher_from/'steps'/name)
    if a.backend!='metal' and (a.attention!='dense' or a.metal_attention_kernel!='steel' or
                             a.metal_attention_dtype!='bf16' or a.metal_tier!='diagnostic' or a.metal_ane!='off' or a.metal_weight_format!='bf16'):
        p.error('native attention policies require --backend metal')
    if a.metal_weight_format=='q8' and a.metal_ane!='off':p.error('Q8 requires GPU-only linears; ANE is not a qualified composition')
    if a.continue_from and (a.reference or extra_inputs):p.error('use a separate continuation/reference benchmark')
    if a.continue_from and a.blocks != 50:p.error('continuation requires all 50 blocks')
    evaluations = CASES[a.case]
    if not 1 <= evaluations <= 6 or (a.case == 'B6' and evaluations != 6):
        p.error('benchmark evaluation contract violated')
    if a.case != 'B1' and (a.blocks != 50 or a.components or a.regions or a.capture_boundaries):
        p.error('block scaling and component fences are B1 diagnostics only')
    if a.trace and a.case != 'B1':
        p.error('Metal System Trace capture is a B1 diagnostic')
    if not 1<=a.trace_window<=60:p.error('trace window must be 1..60 seconds')
    if a.save_conditioning and a.conditioning.exists():
        p.error('conditioning cache already exists; use load-many mode or a new path')
    if not a.save_conditioning and not a.conditioning.is_file():
        p.error('create the conditioning cache once with --save-conditioning')
    if a.save_conditioning and a.blocks != 50:
        p.error('create the conditioning cache with all 50 blocks')
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'record.json').exists() or (a.output/'run.log').exists():
        p.error('output already contains a run; use a fresh output directory')
    a.conditioning = a.conditioning.resolve(); a.conditioning.parent.mkdir(parents=True, exist_ok=True)
    a.binary = a.binary.resolve(); a.model = a.model.resolve()
    # Keep the exact executable even while later experiments rebuild ./bin/h3cli.
    original_binary = a.binary
    a.binary = a.output/'bin/h3cli'
    a.binary.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(original_binary,a.binary)
    env = os.environ.copy()
    env.update(H3_TEST_MAX_EVALUATIONS='6', H3_CPU_SAMPLER='1')
    env.setdefault('H3_DIT_COMMAND_BLOCKS','5')
    # Bind the runtime-loaded shader to the same immutable source snapshot.
    provenance_path=original_binary.parent/'build-provenance.json'
    if provenance_path.is_file():
        frozen=json.loads(provenance_path.read_text())
        env['H3_SHADER_PATH']=str(Path(frozen['source_snapshot'])/'src/metal/shaders.metal')
    env.pop('H3_TEST_NATIVE_QKV',None)
    env.pop('H3_TEST_NATIVE_QKV_BLOCKS',None)
    for name in ('H3_DEBUG_DIT_DIR','H3_DEBUG_DIT_STEPS','H3_TEST_NATIVE_BOUNDARIES','H3_PROFILE_REGIONS','H3_TEST_NATIVE_STEP_DIR','H3_TEST_NATIVE_TEACHER_DIR','H3_TEST_NATIVE_RANGES'):
        env.pop(name,None)
    if a.teacher_from:env['H3_TEST_NATIVE_TEACHER_DIR']=str(a.teacher_from/'steps')
    if a.capture_ranges:env['H3_TEST_NATIVE_RANGES']='1'
    if a.regions:env['H3_PROFILE_REGIONS']='1'
    if a.capture_steps:
        (a.output/'steps').mkdir(exist_ok=True)
        env['H3_TEST_NATIVE_STEP_DIR']=str(a.output/'steps')
    if a.capture_boundaries:
        (a.output/'boundaries').mkdir(exist_ok=True)
        env.update(H3_DEBUG_DIT_DIR=str(a.output/'boundaries'),H3_DEBUG_DIT_STEPS='0',H3_TEST_NATIVE_BOUNDARIES='1')
    if a.capture_qkv:
        if a.case!='B1':p.error('QKV capture is B1-only')
        env['H3_TEST_NATIVE_QKV']=str(a.output/'qkv.bf16')
        env['H3_TEST_NATIVE_QKV_BLOCKS']='0,24,49'
    for name in ('H3_REUSE_STEPS','H3_DIT_ACTIVE_BLOCKS','H3_CORE_REUSE','H3_TOKEN_REDUCTION'):
        if name in env:
            p.error(name+' must be unset for a reference benchmark')
    # CPU Euler gives directly observable completed step boundaries and matches
    # the immutable baseline. It does not move DiT execution off the GPU.
    if a.blocks != 50:
        env['H3_TEST_DIT_BLOCKS'] = str(a.blocks)
    else:
        env.pop('H3_TEST_DIT_BLOCKS', None)
    if a.components:
        env['H3_PROFILE_COMPONENTS'] = '1'
    else:
        env.pop('H3_PROFILE_COMPONENTS', None)
    xcode = Path('/Applications/Xcode.app/Contents/Developer')
    if xcode.exists():
        env['DEVELOPER_DIR'] = str(xcode)
    command = [str(a.binary), '-d', str(a.model), '-p', a.prompt,
        '--backend', a.backend, '--metal-attention', a.attention, '--width', str(a.width),
        '--height', str(a.height), '--frames', str(a.frames), '--steps', str(evaluations),
        '--seed', str(a.seed), '--layers', '50', '--reuse', '1', '--core-reuse', '1',
        '--use-slower-bf16-mlp', '--use-slower-bf16-qkv', '--use-slower-bf16-attention-output',
        '--profile', '--save-av-state', str(a.output/'result.h3av'), '-o', str(a.output/'result.mp4'),
        '--save-conditioning' if a.save_conditioning else '--load-conditioning', str(a.conditioning)]
    if a.schedule_cache:
        command.append('--conditioning-schedule')
    if not a.full_vae:
        command += ['--preview-vae', '--preview-vae-model', str(a.preview_model.resolve())]
    for reference in a.reference:
        command += ['--ref-image', str(reference.resolve())]
    for key,path in extra_inputs.items():command += ['--'+key.replace('_','-'),str(path.resolve())]
    if a.continue_from:
        command += ['--continue-from',str(a.continue_from.resolve()),'--continue-context',str(a.continue_context),'--keep-continuation-prefix']
    metal={key.replace('_','-'):getattr(a,key) for key in ('metal_attention_kernel','metal_attention_dtype','metal_attention_layout','metal_tier','metal_ane','metal_ane_rows','metal_ane_chunk','metal_weight_format','metal_q8_kernel','sol_q_block','sol_kv_block','sol_tau','sol_dense_layers','sol_dense_steps','sol_dense_sigma','sol_local_radius','sol_min_exact')}
    if a.backend=='metal':
        for key,value in metal.items():command += ['--'+key,str(value)]
    record = {'schema': 1, 'case': a.case, 'evaluations': evaluations, 'blocks': a.blocks,
        'geometry': [a.width, a.height, a.frames], 'backend': a.backend, 'attention': a.attention,
        'execution_composition':'hybrid' if a.backend=='metal' else 'mpsgraph',
        'qualification':'unqualified' if a.backend=='metal' else 'reference',
        'metal_options':metal if a.backend=='metal' else None,
        'limits_sha256':sha(ROOT/'tests/metal_native_limits.json'),
        'continuation_sha256':sha(a.continue_from) if a.continue_from else None,
        'continuation_context':a.continue_context if a.continue_from else None,
        'weight_precision': a.metal_weight_format, 'sampler': 'cpu-euler', 'seed': a.seed, 'prompt': a.prompt,
        'reference_sha256': [sha(reference) for reference in a.reference] if a.reference else None,
        'extra_inputs_sha256':{key:sha(path) for key,path in extra_inputs.items()},
        'conditioning': 'save' if a.save_conditioning else 'load',
        'conditioning_sha256_before': sha(a.conditioning) if a.conditioning.exists() else None,
        'preview_vae': not a.full_vae, 'component_fences': a.components, 'region_fences':a.regions,
        'capture_qkv': a.capture_qkv, 'capture_boundaries':a.capture_boundaries,'capture_steps':a.capture_steps,
        'capture_ranges':a.capture_ranges, 'teacher_from':str(a.teacher_from) if a.teacher_from else None,
        'teacher_input_sha256':teacher_hashes,
        'command': command, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'model_path': str(a.model),
        'original_binary': str(original_binary),
        'manifest': manifest(a.binary, a.model, env, original_binary)}
    (a.output/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    begin = time.monotonic(); samples = []; timed_out = False
    trace_child=None;trace_log=None
    with (a.output/'run.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                if a.trace and trace_child is None and 'phase start denoise:' in (a.output/'run.log').read_text():
                    trace_command=['xcrun','xctrace','record','--template','Metal System Trace',
                        '--output',str(a.output/'metal.trace'),'--attach',str(child.pid),
                        '--time-limit',str(a.trace_window)+'s']
                    if a.metal_ane!='off':trace_command+=['--instrument','Neural Engine','--instrument','Core ML','--instrument','os_signpost']
                    record['trace_command']=trace_command
                    trace_log=(a.output/'trace.log').open('w')
                    trace_child=subprocess.Popen(trace_command,cwd=ROOT,env=env,stdout=trace_log,stderr=subprocess.STDOUT)
                sample = command_output(['ps', '-o', 'rss=', '-p', str(child.pid)])
                if sample['returncode'] == 0 and sample['stdout'].isdigit():
                    samples.append({'seconds': time.monotonic()-begin, 'resident_bytes': int(sample['stdout'])*1024})
                if time.monotonic()-begin > a.timeout:
                    timed_out = True; child.terminate(); break
                time.sleep(1)
        finally:
            if child.poll() is None:
                child.terminate()
            try:
                result = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); result = child.wait()
            if trace_child:
                try:record['trace_returncode']=trace_child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    trace_child.terminate()
                    try:trace_child.wait(timeout=10)
                    except subprocess.TimeoutExpired:trace_child.kill();trace_child.wait()
                    record['trace_returncode']=-1
                trace_log.close()
    log = (a.output/'run.log').read_text()
    if a.trace:
        record['trace_pass']=record.get('trace_returncode')==0 and (a.output/'metal.trace').exists()
    steps = [json.loads(row) for row in re.findall(r'h3_step (\{[^\n]+\})', log, re.M)]
    components = [json.loads(row) for row in re.findall(r'h3_component (\{[^\n]+\})', log, re.M)]
    record.update(returncode=result, timed_out=timed_out, wall_seconds=time.monotonic()-begin,
        peak_sampled_resident_bytes=max((s['resident_bytes'] for s in samples), default=None),
        memory_samples=samples, steps=steps, components=components, timing=statistics_for(steps),
        execution=[json.loads(row) for row in re.findall(r'h3_execution (\{[^\n]+\})',log)],
        layout=[json.loads(row) for row in re.findall(r'h3_layout (\{[^\n]+\})',log)],
        recipes=[json.loads(row) for row in re.findall(r'h3_recipe (\{[^\n]+\})',log)],
        weights=[json.loads(row) for row in re.findall(r'h3_weights (\{[^\n]+\})',log)],
        ranges=[json.loads(row) for row in re.findall(r'h3_range (\{[^\n]+\})',log)],
        scores=[json.loads(row) for row in re.findall(r'h3_score (\{[^\n]+\})',log)],
        teacher_steps=[json.loads(row) for row in re.findall(r'h3_teacher (\{[^\n]+\})',log)],
        mixed=[json.loads(row) for row in re.findall(r'h3_mixed (\{[^\n]+\})',log)],
        sol_policy=[json.loads(row) for row in re.findall(r'h3_sol_policy (\{[^\n]+\})',log)],
        sol=[json.loads(row) for row in re.findall(r'h3_sol (\{[^\n]+\})',log)],
        conditioning_sha256_after=sha(a.conditioning) if a.conditioning.exists() else None,
        conditioning_identity=conditioning_identity(a.conditioning),
        av_sha256=sha(a.output/'result.h3av') if (a.output/'result.h3av').exists() else None,
        media_sha256=sha(a.output/'result.mp4') if (a.output/'result.mp4').exists() else None)
    expected = 'native_attention' if a.backend == 'metal' else 'mps_attention'
    record['ane_setup']=[json.loads(row) for row in re.findall(r'h3_ane_setup (\{[^\n]+\})',log)]
    record['ane']=[json.loads(row) for row in re.findall(r'h3_ane (\{[^\n]+\})',log)]
    record['evaluation_contract_pass'] = (result == 0 and not timed_out and
        len(steps) == evaluations and [s['step'] for s in steps] == list(range(1,evaluations+1)) and
        all(s['evaluated'] and s[expected] == a.blocks for s in steps))
    record['conditioning_bypass_pass'] = a.save_conditioning or 'h3cond hit; tokenizer, text/vision and reference encoders skipped' in log
    if a.metal_weight_format=='q8':
        record['q8_execution_pass']=(len(record['weights'])==evaluations and
            [r['step'] for r in record['weights']]==list(range(1,evaluations+1)) and
            all(r['format']=='q8' and r['recipe']==1 and r['group_size']==64 and
                r['linear_path']==('metal-simdgroup-q8' if a.metal_q8_kernel=='simdgroup' else 'metal-dequant-mpsgraph') and r['weights']==4*a.blocks and
                r['q8_dispatches']==4*a.blocks and r['storage_bytes']==r['source_bytes']*17//32 and
                r['activation_dtype']=='bf16' and r['accumulation']==('fp32' if a.metal_q8_kernel=='simdgroup' else 'mpsgraph-internal') for r in record['weights']))
        record['evaluation_contract_pass'] &= record['q8_execution_pass']
    if a.teacher_from:
        record['teacher_input_pass'] = (len(record['teacher_steps']) == evaluations and
            teacher['conditioning_sha256_after'] == record['conditioning_sha256_after'] and
            all(sha(a.teacher_from/'steps'/n)==v for n,v in teacher_hashes.items()) and
            all(sha(a.output/'steps'/f'step-{s+1:03d}-{m}-input.f32') ==
                teacher_hashes[f'step-{s:03d}-{m}-latent.f32'] for s in range(1,evaluations) for m in ('video','audio')))
        record['evaluation_contract_pass'] &= record['teacher_input_pass']
    if a.capture_steps:
        record['step_tensor_sha256']={p.name:sha(p) for p in sorted((a.output/'steps').glob('*.f32'))}
    if a.capture_ranges:
        stages={'residual-input','raw-q','raw-k','raw-v','rope-q','rope-k','value','attention-output',
                'projection-output','attention-residual','mlp-output','residual-output'}
        ranges=record['ranges'];scores=record['scores']
        expected={(s,b) for s in range(1,evaluations+1) for b in range(a.blocks)}
        record['range_capture_pass']=(
            len(ranges)==len(expected)*len(stages) and
            {(r['step'],r['block'],r['stage']) for r in ranges}=={(s,b,k) for s,b in expected for k in stages} and
            all(r['count']>0 and r['nonfinite']==0 for r in ranges) and
            len(scores)==len(expected) and {(r['step'],r['block']) for r in scores}==expected and
            all(r['heads']==56 and r['query_rows_per_head']==16 and r['nonfinite']==0 and
                1<=r['sum_minimum']<=r['sum_maximum']<=r['score_count']/(56*16)*1.00001 for r in scores))
        record['evaluation_contract_pass'] &= record['range_capture_pass']
    if a.backend=='metal' and a.attention=='sol':
        record['sol_policy_pass']=sol_policy_check(record)
        record['sol_routed_blocks']=len(record['sol'])
        record['sol_approximate_pairs']=sum(r['approximate'] for r in record['sol'])
        record['sol_performance_evidence']=record['sol_policy_pass'] and record['sol_approximate_pairs']>0
        record['evaluation_contract_pass'] &= record['sol_policy_pass']
    if a.backend=='metal' and a.metal_attention_dtype=='fp16':
        record['mixed_range_pass'] = (len(record['mixed'])==evaluations*a.blocks and
            {(r['step'],r['block']) for r in record['mixed']}==
            {(s,b) for s in range(1,evaluations+1) for b in range(a.blocks)} and
            all(r['invalid_heads']==0 for r in record['mixed']))
        record['evaluation_contract_pass'] &= record['mixed_range_pass']
    if a.backend=='metal' and a.metal_attention_layout=='fused':
        routed_by_step={s:sum(r['step']==s for r in record['sol']) for s in range(1,evaluations+1)}
        record['layout_pass']=(len(record['layout'])==evaluations and
            [r['step'] for r in record['layout']]==list(range(1,evaluations+1)) and
            all(r['prepares']==a.blocks and r['inplace_packs']==a.blocks-routed_by_step[r['step']] and
                r['scan_replaced_bytes']>0 and r['partial_buffer_bytes_written']*192==r['scan_replaced_bytes']*7
                for r in record['layout']) and
            len(record['recipes'])==evaluations and all(r['layout_fusion']==1 and r['layout_recipe']==1 for r in record['recipes']))
        record['evaluation_contract_pass'] &= record['layout_pass']
    if a.metal_ane!='off':
        records=record['ane']
        record['ane_execution_pass']=(len(records)==evaluations*a.blocks and
            {(r['step'],r['block']) for r in records}=={(s,b) for s in range(evaluations) for b in range(a.blocks)} and
            all(r['ane_rows']>=0 and r['ane_rows']<=a.metal_ane_rows for r in records))
        record['ane_hardware_ready']=bool(record['ane_setup']) and all(r['ready'] and r['ane_matmuls']==r['total_matmuls']>0 for r in record['ane_setup'])
        record['ane_offloaded_blocks']=sum(r['ane_rows']>0 and not r.get('fallback',False) for r in records)
        record['ane_recovered_blocks']=sum(r.get('fallback',False) for r in records)
        record['evaluation_contract_pass'] &= record['ane_execution_pass']
    (a.output/'record.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: record[k] for k in ('case','backend','wall_seconds','timing','returncode','evaluation_contract_pass','conditioning_bypass_pass')}, indent=2))
    if not record['evaluation_contract_pass'] or not record['conditioning_bypass_pass'] or (a.trace and not record['trace_pass']):
        raise SystemExit('Benchmark failed; inspect '+str(a.output/'run.log'))

if __name__ == '__main__':
    main()
