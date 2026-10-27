#!/usr/bin/env python3
"""Run a test with SIGTERM unwinding its cleanup blocks, including child servers."""
from pathlib import Path
import runpy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_process

if __name__=='__main__':
    sys.argv=sys.argv[1:]
    sys.path.insert(0,str(Path(sys.argv[0]).resolve().parent))
    try:
        with build_process.cancellation():runpy.run_path(sys.argv[0],run_name='__main__')
    except KeyboardInterrupt:sys.exit(130)
