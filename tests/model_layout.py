#!/usr/bin/env python3
"""Model-loader regression: either standalone assembled mode may be installed."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path(os.getenv('H3_MODEL_DIR','models/MiniMax-H3')))
    args=p.parse_args();model=args.model.resolve(strict=True)
    for mode in ('FL2VA','Ref2VA'):assert (model/mode/'transformer/config.json').is_file()
    with tempfile.TemporaryDirectory(prefix='h3-model-layout-') as temp:
        root=Path(temp)
        def run(name,options,expected=None):
            cmd=['./bin/h3cli','-d',str(root/name),*options]
            result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
            if expected:
                assert result.returncode and expected in result.stdout,result.stdout
            else:assert result.returncode==0,result.stdout
        for name,modes in [('fl',('FL2VA',)),('ref',('Ref2VA',)),('both',('FL2VA','Ref2VA'))]:
            (root/name).mkdir()
            for mode in modes:(root/name/mode).symlink_to(model/mode,target_is_directory=True)
            run(name,['--info'])
        small=['-p','A woman smiles.','--width','128','--height','128','--frames','22','--steps','2','-o',str(root/'must-not-exist.mp4')]
        run('ref',small,'require the FL2VA checkpoint')
        run('ref',small+['--first-frame','inputs/face1.jpg'],'require the FL2VA checkpoint')
        run('fl',small+['--ref-image','inputs/face1.jpg'],'require the Ref2VA checkpoint')
        assert not (root/'must-not-exist.mp4').exists()
        # A present but broken mode must not silently borrow the other weights.
        broken=root/'broken';broken.mkdir();(broken/'Ref2VA').symlink_to(model/'Ref2VA',target_is_directory=True)
        (broken/'FL2VA/transformer').mkdir(parents=True);(broken/'FL2VA/transformer/config.json').write_text('{}')
        run('broken',['--info'],'missing required model file')
    print('ok: FL2VA-only, Ref2VA-only, both, missing selected mode and incomplete installation')

if __name__=='__main__':main()
