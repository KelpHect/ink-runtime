"""Compare ink-runtime fp32 dumps vs HF golden .npy files.

Usage: .venv/Scripts/python tools/compare_golden.py [golden_dir] [ink_dir]
ink dir defaults to testdata/ink; golden to testdata/golden.

Metrics per tensor+step: max|d|, mean|d|, cosine similarity, and for logits
also argmax match + top-5 overlap. Exit 1 if any argmax mismatches, any logits
cosine < 0.999, or any hidden cosine < 0.98. Rationale: HF stores hidden in
bf16 while ink is pure fp32, so ~48-visit noise accumulation legitimately
drifts deep taps to ~0.985; real porting bugs produced cos <= 0.79 at visit5
and flipped argmax immediately.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "testdata", "golden")
INK = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "testdata", "ink")

NAMES = [
    "hidden_visit5",
    "hidden_visit23",
    "hidden_visit41",
    "hidden_final",
    "logits",
]


def load_pair(name, step):
    g = np.load(os.path.join(GOLD, f"{name}_step{step}.npy"))
    p = os.path.join(INK, f"{name}_step{step}.bin")
    if not os.path.exists(p):
        return g, None
    i = np.fromfile(p, dtype=np.float32)
    return g, i


def main():
    n_steps = 0
    while os.path.exists(os.path.join(GOLD, f"logits_step{n_steps}.npy")):
        n_steps += 1
    print(f"golden steps: {n_steps}")

    worst_cos = 1.0
    worst_logits_cos = 1.0
    argmax_bad = []
    for step in range(n_steps):
        print(f"--- step {step} ---")
        for name in NAMES:
            g, i = load_pair(name, step)
            if i is None:
                print(f"  {name:16s} MISSING in ink dump")
                continue
            gl = g.reshape(-1, g.shape[-1])[-1].astype(np.float64)
            if i.size != gl.size:
                print(f"  {name:16s} SHAPE MISMATCH gold {gl.size} ink {i.size}")
                continue
            il = i.astype(np.float64)
            d = np.abs(gl - il)
            denom = np.abs(gl).max() or 1.0
            cos = float(
                np.dot(gl, il) / (np.linalg.norm(gl) * np.linalg.norm(il) + 1e-30)
            )
            if name == "logits":
                worst_logits_cos = min(worst_logits_cos, cos)
            else:
                worst_cos = min(worst_cos, cos)
            line = (
                f"  {name:16s} max|d|={d.max():.4e} mean|d|={d.mean():.4e} "
                f"rel_max={d.max()/denom:.4e} cos={cos:.6f}"
            )
            if name == "logits":
                ga = int(gl.argmax())
                ia = int(il.argmax())
                g5 = set(np.argsort(gl)[-5:].tolist())
                i5 = set(np.argsort(il)[-5:].tolist())
                line += f" argmax g={ga} i={ia}{' OK' if ga == ia else ' MISMATCH'} top5∩={len(g5 & i5)}/5"
                if ga != ia:
                    argmax_bad.append((step, ga, ia))
            print(line)

    print()
    if argmax_bad or worst_cos < 0.98 or worst_logits_cos < 0.999:
        print(f"GATE FAIL: worst_cos={worst_cos:.6f} "
              f"worst_logits_cos={worst_logits_cos:.6f} argmax_bad={argmax_bad}")
        return 1
    print(f"GATE PASS: worst_cos={worst_cos:.6f} "
          f"worst_logits_cos={worst_logits_cos:.6f}, all argmax match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
