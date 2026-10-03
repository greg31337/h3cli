#!/usr/bin/env python3
"""Metal verbose recognition is opt-in; portable CUDA remains quiet.

Both backends preserve IDs and decoding under the diagnostic environment flag.
"""
import json
import os
import platform
from pathlib import Path
import subprocess
import tempfile

from tokenizer_reference import H3, ROOT

with tempfile.TemporaryDirectory(prefix='h3-tokenizer-') as tmp:
    root = Path(tmp)
    (root/'tokenizer.json').write_text(json.dumps({'normalizer': {'type': 'NFC'},
        'model': {'type': 'BPE', 'unk_token': None, 'vocab': {}, 'merges': []}}))
    (root/'corpus.json').write_text(json.dumps([{'name': 'all', 'text': ''.join(H3)}]))
    results = []
    for verbose in (None, '0', '1'):
        env = os.environ.copy()
        env.pop('H3_DEBUG_TOKENIZER', None)
        if verbose is not None:
            env['H3_DEBUG_TOKENIZER'] = verbose
        result = subprocess.run([str(ROOT/'bin/tokenizer_dump'), str(root/'tokenizer.json'),
            str(root/'corpus.json')], env=env, capture_output=True, text=True, check=True)
        results.append(result)
    assert results[0].stdout == results[1].stdout == results[2].stdout
    assert results[0].stderr == results[1].stderr == ''
    expected = [f'special token: "{text}" -> {151669+i}' for i, text in enumerate(H3)] if platform.system()=="Darwin" else []
    assert results[2].stderr.splitlines() == expected
    assert json.loads(results[0].stdout)[0]['ids'] == list(range(151669, 151676))
print('ok: tokenizer diagnostics are opt-in; IDs/decoding unchanged')
