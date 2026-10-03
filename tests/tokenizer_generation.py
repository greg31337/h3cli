#!/usr/bin/env python3
from source_tree import source_path
"""Before/after H3 token-fix renders on local released weights.

All conditioning is freshly encoded. Only isolated source copies are instrumented
to capture IDs, visual embeddings, reference conditions and final AV latents.
No conditioning replay, reduced layer counts, or precision flags.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from source_tree import build_path
import shutil
import subprocess
import tarfile
import time

from continuation_regression import HELPER, instrument

from source_tree import copy_source_tree

ROOT = Path(__file__).resolve().parents[1]
CONTROL = 'A woman looks at the camera and slowly walks across the room.'
DIALOGUE = 'A woman looks at the camera and says:\n<d>[English] Where are you going?</d>'
CUTOFF_BASE = "A woman looks at the camera and says:\n<d>[English] Wait, don't—</d>"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(directory):
    instrument(directory)
    path = source_path(directory, 'engine.c')
    source = path.read_text()
    # Remove all replay calls. Both builds must actually encode their own IDs.
    source = '\n'.join(line for line in source.split('\n')
                       if not ('regression_buffer(' in line and line.endswith(',1);')))
    needle = '    h3_layout_spec spec ='
    source = source.replace(needle, '''    if (text.diagnostics) {
        regression_buffer("ids",text.diagnostics->ids,text.diagnostics->tokens*4,0);
        regression_buffer("positions",text.diagnostics->positions,text.diagnostics->tokens*3*4,0);
        regression_buffer("spans",text.diagnostics->spans,text.diagnostics->span_count*2*8,0);
    } else regression_buffer("ids",ids,token_count*4,0);
'''+needle)
    path.write_text(source)
    path = source_path(directory, 'multimodal.c')
    source = path.read_text().replace('#define H3_VISION_START', HELPER+'\n#define H3_VISION_START')
    needle = '    int ok = h3_text_encode_multimodal_bf16('
    assert source.count(needle) == 1
    source = source.replace(needle, '''    for (size_t i=0; i<span_count; i++) {
        char name[96]; snprintf(name,sizeof(name),"vision-%zu-merged",i);
        regression_buffer(name,(void *)spans[i].embeddings,spans[i].tokens*H3_VISION_OUTPUT_WIDTH*2,0);
        for (size_t k=0; k<3; k++) {
            snprintf(name,sizeof(name),"vision-%zu-deep-%zu",i,k);
            regression_buffer(name,(void *)spans[i].deepstack[k],spans[i].tokens*H3_VISION_OUTPUT_WIDTH*2,0);
        }
    }
'''+needle)
    path.write_text(source)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='e4474dd')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/tokenfix-validation/generation')
    p.add_argument('--only', default='control,dialogue,cutoff,cutoff-unmarked,ref-dialogue')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    builds = {}
    for label in ('before', 'after'):
        directory = out/label
        if not (args.resume and (build_path(directory, 'h3cli')).exists()):
            directory.mkdir(exist_ok=True)
            if label == 'before':
                archive = out/'baseline.tar'
                subprocess.run(['git', 'archive', '-o', str(archive), args.baseline], cwd=ROOT, check=True)
                with tarfile.open(archive) as f:
                    f.extractall(directory, filter='data')
            else:
                copy_source_tree(ROOT, directory)
            prepare(directory)
            with (directory/'build.log').open('w') as log:
                subprocess.run(['make', '-j8', 'all'], cwd=directory, stdout=log, stderr=log, check=True)
        builds[label] = directory
    if args.prepare_only:
        return
    # A silent, known image-derived video exercises the released video-reference
    # path without supplying reference speech that could confound the test.
    reference = out/'body-reference.mp4'
    if not reference.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-framerate', '24',
            '-i', str(ROOT/'inputs/body1.jpg'), '-vf', 'scale=256:256', '-frames:v', '72',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(reference)], check=True)
    face = ['--first-frame', str(ROOT/'inputs/face1.jpg')]
    cases = {'control': (CONTROL, [], ('before', 'after')),
             'dialogue': (DIALOGUE, face, ('before', 'after')),
             'cutoff': (CUTOFF_BASE+'<|cutoff|>', face, ('after',)),
             'cutoff-unmarked': (CUTOFF_BASE, face, ('after',)),
             'ref-dialogue': (DIALOGUE, ['--ref-image', str(ROOT/'inputs/face1.jpg'),
                 '--ref-silent-video', str(reference)], ('before', 'after'))}
    records_path = out/'results.json'
    records = json.loads(records_path.read_text()) if args.resume and records_path.exists() else {}
    for case in args.only.split(','):
        prompt, references, versions = cases[case]
        for label in versions:
            name = case+'-'+label
            if args.resume and name in records:
                continue
            directory = builds[label]
            base = out/name
            cmd = [str(build_path(directory, 'h3cli')), '-d', str(ROOT/'models/MiniMax-H3'), '-p', prompt,
                '--width', '256', '--height', '256', '--frames', '124', '--steps', '20',
                '--seed', '72', *references, '-o', str(base)+'.mp4']
            env = {k: v for k, v in os.environ.items() if not k.startswith('H3_')}
            env['H3_REGRESSION_DUMP'] = str(base)
            print('render', name, flush=True)
            start = time.monotonic()
            with Path(str(base)+'.log').open('w') as log:
                subprocess.run(cmd, cwd=directory, env=env, stdout=log, stderr=log, check=True)
            hashes = {}
            for f in out.glob(name+'.*'):
                suffix = f.name[len(name)+1:]
                if f.is_file() and (suffix in ('ids', 'positions', 'spans', 'text', 'video', 'audio',
                                                'mp4', 'condition-video', 'condition-audio') or
                                    suffix.startswith('vision-')):
                    hashes[suffix] = sha(f)
            import struct
            data = Path(str(base)+'.ids').read_bytes()
            records[name] = {'command': cmd, 'prompt': prompt, 'seconds': time.monotonic()-start,
                             'hashes': hashes, 'ids': list(struct.unpack('<'+'I'*(len(data)//4), data)),
                             'binary_sha256': sha(build_path(directory, 'h3cli'))}
            records_path.write_text(json.dumps(records, indent=2)+'\n')
            print('completed', name, round(records[name]['seconds'], 1), 'seconds', flush=True)


if __name__ == '__main__':
    main()
