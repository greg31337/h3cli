#!/usr/bin/env python3
"""Model-free preflight failures, including order-independent SOL controls."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import os,subprocess
base=['./bin/h3cli','-d','/nonexistent-h3-sol-model','-p','test','--steps','2']
env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
cases=[(['--cuda-attention','sol','--sol-q-block','16'],'invalid CUDA SOL'),
       (['--sol-q-block','16','--cuda-attention','sol'],'invalid CUDA SOL'),
       (['--cuda-attention','sol','--sol-kv-block','128'],'invalid CUDA SOL'),
       (['--cuda-attention','sol','--sol-min-exact','nan'],'invalid SOL minimum exact fraction'),
       (['--cuda-attention','sol','--core-reuse','2'],'reuse/core-reuse'),
       (['--cuda-attention','sol','--backend','metal'],'requires CUDA'),
       (['--cuda-attention','sol','--decode-av-state','missing'],'decode'),
       (['--cuda-attention','invalid'],'cuda-attention')]
for flags,message in cases:
    p=subprocess.run(base+flags,env=env,capture_output=True,text=True,timeout=15)
    assert p.returncode!=0,(flags,p.stderr)
    # Numeric parsing and platform capability errors can precede policy errors.
    assert any(s in p.stderr.lower() for s in [message.lower(),'invalid cuda sol','requires cuda','not compatible','cannot combine','use cuda options on linux']), (flags,p.stderr)
print('PASS CUDA SOL CLI preflight/order checks')
