#!/usr/bin/env python3
"""Select independent bounded workflow checks without resetting the shared ledger."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('cases',nargs='*')
p.add_argument('--model',default=os.environ.get('H3_MODEL_DIR','models/MiniMax-H3'))
p.add_argument('--cache',default=os.environ.get('H3_TEST_QUANT_CACHE','outputs/quant-5090/packed'))
a=p.parse_args()
cases=[('state-fp8','fp8','state'),('state-nvfp4','nvfp4','state'),
       ('continuation-nvfp4','nvfp4','continuation'),('continuation-fp8','fp8','continuation'),
       ('warm288-off','off','warm288'),('warm288-fp8','fp8','warm288'),('warm288-nvfp4','nvfp4','warm288'),
       ('warm480-nvfp4','nvfp4','warm480'),('warm480-fp8','fp8','warm480'),('warm480-off','off','warm480'),
       ('cancel-nvfp4','nvfp4','cancel'),('cancel-fp8','fp8','cancel'),('pressure-fp8','fp8','pressure')]
out=Path('outputs/quant-5090')
for name,mode,kind in cases:
    if a.cases and name not in a.cases:continue
    record=out/(name+'.json')
    if record.exists():
        assert json.loads(record.read_text())['status']=='passed',f'{name}: inspect previous failure before repeating'
        continue
    directory=out/name;directory.mkdir(exist_ok=True)
    command=['./bin/quant_workflow',str(Path(a.model).expanduser()),str(Path(a.cache).expanduser()),str(directory),mode,kind]
    result=subprocess.run([sys.executable,'tests/quant_run.py','--name',name,'--kind','render','--',*command])
    # Failures remain in the ledger; unrelated tests can continue. Reservation
    # refusal exits before launching a process and stops this selection.
    if not record.exists():break
    if result.returncode:print('FAILED:',name,flush=True)
