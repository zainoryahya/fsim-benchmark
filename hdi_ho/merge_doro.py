"""Merge the Dorothea result parts into the 5-repeat pickles read by analyze_doro.py and the paper.

* wrapper, wrapper-H, cap, EMT and other-classifier results: repeats 0-1 (results_doro_<x>/dorothea_r2.pkl) and
  repeats 2-4 (results_<x>_dx/dorothea.pkl, written by xqueue2.sh) -> results_doro_<x>/dorothea.pkl (weights 2/5, 3/5);
* threshold and Tomek results: results_doro_{thr,tomek}_r2.pkl + _x.pkl -> results_doro_{thr,tomek}.pkl;
* further baselines: repeats 0-2 (results_doro_base_a, R0=0 NREP=3) and 3-4 (results_doro_base_b, R0=3 NREP=2)
  -> results_doro_base/dorothea.pkl (weights 3/5, 2/5).
Same rules as merge_x.py (metrics are per-repeat means; lists are concatenated in repeat order).
Usage (from hdi_ho): python merge_doro.py [--check]   (--check compares with the existing merged files, writes nothing)
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hdi"))
import merge_x

CHECK = "--check" in sys.argv
P = lambda f: pickle.load(open(f, "rb"))


def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, float):
        return abs(a - b) <= 1e-12 * max(1.0, abs(a))
    return a == b


def out(m, f):
    if CHECK:
        print(f"{f:45s}", "identical" if same(m, P(f)) else "DIFFERENT")
    else:
        pickle.dump(m, open(f, "wb")); print("wrote", f)


def run(a, b, wa, wb, nrep=None):
    merge_x.WA, merge_x.WB = wa, wb
    merge_x.UNMERGED = set()
    m = merge_x.merge(a, b)
    if nrep:
        m["unmerged"] = sorted(merge_x.UNMERGED); m["nrep"] = nrep
    return m


for x in ("wrap", "wrapH", "cap", "emt", "xclf"):
    d = f"results_doro_{x}"
    out(run(P(f"{d}/dorothea_r2.pkl"), P(f"results_{x}_dx/dorothea.pkl"), 2 / 5, 3 / 5), f"{d}/dorothea.pkl")
for x in ("thr", "tomek"):
    out(run(P(f"results_doro_{x}_r2.pkl"), P(f"results_doro_{x}_x.pkl"), 2 / 5, 3 / 5), f"results_doro_{x}.pkl")
m = run(P("results_doro_base_a/dorothea.pkl"), P("results_doro_base_b/dorothea.pkl"), 3 / 5, 2 / 5)
m["nrep"] = 5
out(m, "results_doro_base/dorothea.pkl")
