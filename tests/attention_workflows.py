"""Read-only historical metrics; rendering uses cuda_single_run.py."""
import argparse
import fcntl
import hashlib
import html
import json
import re
from pathlib import Path
import subprocess
from attention_run import run,sha,profile
from continuation_metrics import state,prefix_bytes
from attention_media import inspect_state_media

def compare_av(reference,candidate):
    import numpy as np
    ga,va,aa=state(reference);gb,vb,ab=state(candidate)
    assert ga==gb,'AV geometry differs'
    result={}
    for name,left,right in [('video',va,vb),('audio',aa,ab)]:
        x=np.frombuffer(left,dtype='<f4').astype('float64');y=np.frombuffer(right,dtype='<f4').astype('float64')
        d=y-x;xn=np.linalg.norm(x);yn=np.linalg.norm(y)
        result[name]={'exact':left==right,'finite':bool(np.isfinite(y).all()),
            'relative_l2':float(np.linalg.norm(d)/max(xn,1e-30)),
            'rmse':float(np.sqrt(np.mean(d*d))),'max_abs':float(np.max(np.abs(d))),
            'cosine':float(x@y/max(xn*yn,1e-30))}
    return result
