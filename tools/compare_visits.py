"""Per-visit bisection: ink visitNN_{in,out} f32 bins vs golden visits/*.npy
(last position of step0). Prints cos + max|d| per visit; first visit whose
INPUT already diverges localizes the buggy span."""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(REPO, "testdata", "golden", "visits")
INK = os.path.join(REPO, "testdata", "ink")


def pair(name):
    g = np.load(os.path.join(GOLD, f"{name}.npy")).reshape(-1, 2304)[-1]
    p = os.path.join(INK, f"{name}_step0.bin")
    if not os.path.exists(p):
        return g, None
    return g, np.fromfile(p, dtype=np.float32)


def stat(g, i):
    d = np.abs(g.astype(np.float64) - i.astype(np.float64))
    cos = float(np.dot(g, i) / (np.linalg.norm(g) * np.linalg.norm(i) + 1e-30))
    return d.max(), cos


def main():
    print(f"{'visit':>5} {'in_cos':>9} {'in_maxd':>10} {'out_cos':>9} {'out_maxd':>10}")
    first_bad_in = None
    for v in range(48):
        gi, ii = pair(f"visit{v:02d}_in")
        go, io = pair(f"visit{v:02d}_out")
        md_i = cos_i = md_o = cos_o = float("nan")
        if ii is not None:
            md_i, cos_i = stat(gi, ii)
        if io is not None:
            md_o, cos_o = stat(go, io)
        flag = ""
        if first_bad_in is None and cos_i == cos_i and cos_i < 0.99:
            first_bad_in = v
            flag = "  <-- first bad input"
        print(f"{v:>5} {cos_i:9.4f} {md_i:10.3e} {cos_o:9.4f} {md_o:10.3e}{flag}")
    print()
    if first_bad_in is not None:
        print(f"first diverging visit input: {first_bad_in}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
