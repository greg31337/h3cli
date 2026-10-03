#!/usr/bin/env python3
"""Assemble shared model assets or record optional continuation provenance."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from fold_lora import FoldError, MANIFEST, read_json, sha256


def assemble(base, model, mode):
    base, model = Path(base).resolve(strict=True), Path(model).resolve(strict=True)
    if base == model or base in model.parents or model in base.parents:
        raise FoldError('assembled model and original base must be separate trees')
    transformer = model/mode/'transformer'
    if not (transformer/MANIFEST).is_file():
        raise FoldError(f'fold the selected transformer first: {transformer}')
    links = [(p, model/mode/p.name) for p in (base/mode).iterdir() if p.name != 'transformer']
    links += [(p, model/p.name) for p in base.iterdir() if p.is_file()]
    for source, target in links:
        if os.path.lexists(target) and target.resolve() != source.resolve():
            raise FoldError(f'refusing to replace existing model asset: {target}')
    for source, target in links:
        if not os.path.lexists(target):
            target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    return model


def record_provenance(state, model, mode):
    state, model = Path(state).resolve(strict=True), Path(model).resolve(strict=True)
    with state.open('rb') as stream:
        if stream.read(8) != b'H3AV\r\n\x1a\n':
            raise FoldError('provenance requires a complete .h3av continuation state')
    manifest_path = model/mode/'transformer'/MANIFEST
    manifest = read_json(manifest_path.read_bytes())
    data = dict(schema_version=1, state_sha256=sha256(state), model_directory=str(model), mode=mode,
                manifest_sha256=sha256(manifest_path), base_sha256=manifest['base_sha256'],
                adapters=[{k: a[k] for k in ('path','sha256','user_scale','profile')} for a in manifest['adapters']],
                note='Informational provenance only. AV continuation compatibility remains the VAE/latent contract; '
                     'full sampler resume still requires the exact model fingerprint.')
    target = Path(str(state)+'.lora.json')
    fd, name = tempfile.mkstemp(prefix=target.name+'.tmp-', dir=state.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(name,target)
    finally:
        if os.path.exists(name): os.unlink(name)
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('assemble')
    a.add_argument('--base',required=True)
    for command in (a,sub.add_parser('provenance')):
        command.add_argument('--model',required=True)
        command.add_argument('--mode',choices=['FL2VA','Ref2VA'],required=True)
    sub.choices['provenance'].add_argument('--av-state',required=True)
    args=p.parse_args()
    try:
        if args.command=='assemble': print(assemble(args.base,args.model,args.mode))
        else: print(record_provenance(args.av_state,args.model,args.mode))
    except (FoldError,OSError) as exc:
        p.exit(1,f'workflow: error: {exc}\n')


if __name__=='__main__': main()
