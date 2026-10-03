#!/usr/bin/env python3
"""Runtime LoRA CLI validation; --info uses a local model/device, no folding."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
from pathlib import Path
import subprocess
import tempfile

model=Path(os.getenv('H3_MODEL_DIR','models/MiniMax-H3')).resolve()
adapter=Path(os.getenv('H3_LORA_ADAPTER','lora/downloads/minimax_h3_turbo_v4_step600_ema.safetensors')).resolve()
with tempfile.TemporaryDirectory(prefix='h3-lora-cli-') as tmp:
    root=Path(tmp);cache=root/'must-not-exist'
    def run(args,success=False,contains=None):
        r=subprocess.run(['./bin/h3cli','-d',str(model),*args],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=45,
            env={**os.environ,'XDG_CACHE_HOME':str(root/'xdg')})
        assert (r.returncode==0)==success,r.stdout
        if contains:assert contains in r.stdout,r.stdout
        assert not cache.exists(),r.stdout
        assert not (root/'xdg').exists(),r.stdout
        return r.stdout
    run(['--info'],success=True)
    run(['--info','--lora',str(adapter),'--lora-cache',str(cache),'--lora-memory-mib','1'],success=True,contains='Runtime LoRA: 1 adapter(s), 1 MiB')
    run(['--info','--lora',str(adapter)+':0.5','--lora',str(adapter)+':-1','--lora-cache',str(cache)],success=True,contains='2 adapter(s)')
    run(['--info','--lora-cache',str(cache)],contains='require --lora')
    run(['--info','--lora-memory-mib','1'],contains='require --lora')
    for value in ['0','-1','1.0','1e3','x','18446744073709551615']:
        run(['--info','--lora',str(adapter),'--lora-memory-mib',value],contains='positive integer')
    for value in ['nan','inf','1e100','bad','']:
        run(['--info','--lora',str(adapter)+':'+value],contains='invalid LoRA')
    run(['--info','--lora',str(adapter),'--lora-cache',''],contains='invalid LoRA')
    for flags in [['--lora',str(adapter)],['--lora-cache',str(cache)],['--lora-memory-mib','1']]:
        run(['--decode-av-state',str(root/'missing.h3av'),*flags],contains='decode-only accepts')
    output=run(['--resume-sampler-state',str(root/'missing.h3sample'),'--lora',str(adapter),'--lora-cache',str(cache)])
    assert 'generation-changing' not in output,output
    run(['--resume-sampler-state',str(root/'missing.h3sample'),'--lora',str(adapter),'--steps','2'],contains='generation-changing')
    # Missing selected modes must fail before any fold/cache creation.
    for mode,extra,message in [('Ref2VA',[],'require the FL2VA checkpoint'),('FL2VA',['--ref-image','inputs/face1.jpg'],'require the Ref2VA checkpoint')]:
        stand=root/mode;stand.mkdir();(stand/mode).symlink_to(model/mode,target_is_directory=True)
        r=subprocess.run(['./bin/h3cli','-d',str(stand),'--lora',str(adapter),'--lora-cache',str(cache),'-p','A sunny park.','--width','128','--height','128','--frames','22','--steps','2',*extra],text=True,capture_output=True,timeout=45)
        assert r.returncode and message in r.stderr,r.stderr
        assert not cache.exists()
print('ok: 24 runtime LoRA CLI validation/model-selection cases; no cache created')
