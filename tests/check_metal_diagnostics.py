#!/usr/bin/env python3
"""Independent expected values for bin/metal_diagnostics' GPU probe output."""
import json
import math
from pathlib import Path
import re
import sys

log=Path(sys.argv[1]).read_text()
ranges=[json.loads(x) for x in re.findall(r'h3_range (\{[^\n]+\})',log)]
scores=[json.loads(x) for x in re.findall(r'h3_score (\{[^\n]+\})',log)]
assert len(ranges)==4 and len(scores)==2
for stage,value in (('raw-q',.5),('raw-k',-1.),('raw-v',65536.)):
    row=next(r for r in ranges if r['step']==1 and r['stage']==stage)
    assert row['minimum']==row['maximum']==value and row['count']==33*128
    assert row['nonfinite']==0 and row['below_half_normal']==0
    assert row['over_fp16']==(33*128 if stage=='raw-v' else 0)
r=scores[0]
assert r['step']==1 and r['heads']==1 and r['query_rows_per_head']==16
assert r['score_count']==16*33 and r['nonfinite']==r['zero_exp']==r['below_half_normal_exp']==0
assert math.isclose(r['minimum'],math.sqrt(8),abs_tol=1e-6) and math.isclose(r['maximum'],math.sqrt(8),abs_tol=1e-6)
assert r['minimum_exp']==1 and r['sum_minimum']==r['sum_maximum']==33
assert ranges[-1]['nonfinite']==1 and scores[-1]['nonfinite']>0
print('PASS: strided raw Q/K/V ranges, overflow counts, independent score/softmax, nonfinite rejection')
