"""Test-only hook bootstrap inherited by SGLang's spawned worker."""
import os
if os.environ.get("H3_SGLANG_CAPTURE"):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import cuda_sglang_capture
    cuda_sglang_capture.install()
