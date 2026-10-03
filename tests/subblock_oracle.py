"""Independent NumPy BF16-pooling/router and FP64 masked-softmax oracle.

The tolerances are the predeclared recipe contract; this does not modify or
replace any recorded CUDA reference regression artifact.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def bf16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(x):
    return (np.asarray(x, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def pool(x, scale):
    s, h, d = x.shape
    cells = (s + 63) // 64 * 4
    result = np.zeros((h, cells, d), np.float32)
    for cell in range((s + 15) // 16):
        values = x[cell * 16:min(s, cell * 16 + 16)].astype(np.float64)
        result[:, cell] = from_bf16(bf16(values.mean(axis=0).astype(np.float32) * np.float32(scale)))
    return result


def lse(x, axis):
    maximum = x.max(axis=axis, keepdims=True)
    return (maximum + np.log2(np.exp2(x - maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


def validate(base, sequence, heads, sparsity, force=False, kind=0, qkv=None, tiled=False):
    n = sequence * heads * 128
    if qkv:
        values = from_bf16(np.fromfile(qkv, np.uint16)).reshape(3, sequence, heads, 128)
        q, k, v = values
    else:
        q, k, v = [from_bf16(np.fromfile(str(base) + f".{name}.bf16", np.uint16)).reshape(sequence, heads, 128)
                   for name in ("q", "k", "v")]
    blocks = (sequence + 63) // 64
    keep = min(blocks, 8 * int(np.ceil((1 - float(np.float32(sparsity))) * blocks / 8)))
    scale = np.float32(1 / np.sqrt(np.float32(128)))
    qp, kp = pool(q, scale * np.float32(np.log2(np.e))), pool(k, 1)
    expected_scores = np.empty((heads, blocks, blocks), np.float64)
    valid = (sequence + 15) // 16
    for h in range(heads):
        dots = qp[h].astype(np.float64) @ kp[h].astype(np.float64).T
        for i in range(blocks):
            for j in range(blocks):
                cell = dots[i*4:min(i*4+4, valid), j*4:min(j*4+4, valid)]
                expected_scores[h, i, j] = lse(lse(cell, 1), 0) * np.log(2)
    if tiled:
        assert force, "tiled fixture must retain all keys"
        scores=expected_scores.copy();routes=np.ones((heads,blocks,blocks),np.uint8)
    else:
        scores = np.fromfile(str(base) + ".scores", np.float32).reshape(heads, blocks, blocks)
        routes = np.fromfile(str(base) + ".routes", np.uint8).reshape(heads, blocks, blocks)
    tolerance = 2e-4 + 2e-5 * np.abs(expected_scores)
    assert np.all(np.abs(scores - expected_scores) <= tolerance), "pooled score mismatch"
    assert np.all((routes == 0) | (routes == 1))
    protected = np.zeros(blocks, bool)
    if kind == 4:
        protected[[0, -1]] = True
    for h in range(heads):
        for i in range(blocks):
            order = np.lexsort((np.arange(blocks), -scores[h, i]))
            expected = protected.copy()
            expected[order[:keep]] = True
            if force or protected[i]:
                expected[:] = True
            assert np.array_equal(routes[h, i], expected), "selection/tie/protection mismatch"
            independent = np.lexsort((np.arange(blocks), -expected_scores[h, i]))
            if keep < blocks and expected_scores[h, i, independent[keep-1]] - expected_scores[h, i, independent[keep]] > 2*tolerance[h, i].max():
                expected[protected] = True
                chosen = protected.copy(); chosen[independent[:keep]] = True
                if force or protected[i]: chosen[:] = True
                assert np.array_equal(routes[h, i], chosen), "independent route mismatch"
    row = from_bf16(np.fromfile(str(base) + ".row.bf16", np.uint16)).reshape(sequence, heads, 128)
    head = from_bf16(np.fromfile(str(base) + ".head.bf16", np.uint16)).reshape(heads, sequence, 128).transpose(1, 0, 2)
    assert np.array_equal(row, head), "output layout mismatch"
    assert np.isfinite(row).all()
    # Complete small fixtures, bounded rows/heads for large real captures.
    rows = np.arange(sequence) if sequence <= 129 else np.unique(np.linspace(0, sequence-1, 24).astype(int))
    chosen_heads = np.arange(heads) if heads <= 4 else np.array([0, heads//2, heads-1])
    squared = norm = maximum = 0.0
    for h in chosen_heads:
        for i in rows:
            visible = np.repeat(routes[h, i//64].astype(bool), 64)[:sequence]
            logits = k[visible, h].astype(np.float64) @ q[i, h].astype(np.float64) * float(scale)
            weights = np.exp(logits - logits.max()); weights /= weights.sum()
            oracle = weights @ v[visible, h].astype(np.float64)
            delta = row[i, h].astype(np.float64) - oracle
            squared += float(delta @ delta); norm += float(oracle @ oracle)
            maximum = max(maximum, float(np.max(np.abs(delta))))
    relative = np.sqrt(squared / max(norm, 1e-30))
    assert relative <= .01 and maximum <= .0625, (relative, maximum)
    if kind == 1: assert np.all(row == .5), "constant-value invariant"
    return {"relative_l2": relative, "max_abs": maximum, "max_score_error": float(np.max(np.abs(scores-expected_scores))),
            "density": float(routes.mean()), "compared_rows": int(len(rows)), "compared_heads": int(len(chosen_heads)), "routing_checked": not tiled}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("base", type=Path); p.add_argument("sequence", type=int); p.add_argument("heads", type=int)
    p.add_argument("sparsity", type=float); p.add_argument("--force", action="store_true"); p.add_argument("--kind", type=int, default=0); p.add_argument("--qkv", type=Path)
    p.add_argument("--tiled",action="store_true")
    a = p.parse_args(); print(json.dumps(validate(a.base, a.sequence, a.heads, a.sparsity, a.force, a.kind, a.qkv, a.tiled), indent=2))
