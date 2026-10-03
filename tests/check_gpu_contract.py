#!/usr/bin/env python3
"""Ensure the link test covers every declared backend entry point."""
from pathlib import Path
import re
header=Path("src/gpu.h").read_text()
declared=set(re.findall(r"\b(h3_(?:gpu|qwen)_[a-zA-Z0-9_]+)\s*\(",header))
covered=set(re.findall(r"\)\s*(h3_(?:gpu|qwen)_[a-zA-Z0-9_]+)\s*,",Path("tests/test_gpu_contract.c").read_text()))
assert declared==covered, {"missing":sorted(declared-covered),"extra":sorted(covered-declared)}
print(f"ok: contract covers all {len(declared)} GPU declarations")
