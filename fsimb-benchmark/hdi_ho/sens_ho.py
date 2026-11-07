"""Sensitivity analysis of the pre-registered held-out tests (H1-H6) without the chowdary and singh families
(used in an early prototype run) and without nakayama (class rule misapplied). Reuses analyze_ho.py's run().
Two repeats (as registered: results_*_r2) and five repeats (merged results). Run from hdi_ho.
"""
import contextlib, io, pickle, json
from collections import defaultdict
import numpy as np

src = open("analyze_ho.py").read().split('S = {"n": len(names), "all": run(names, "ALL")}')[0]
res = {}
for suf in ("_r2", ""):
    code = src
    for d in ("results_wrap", "results_wrapH", "results_cap", "results_emt", "results_xclf", "results_ridge"):
        code = code.replace(f'"{d}"', f'"{d}{suf}"')
    g = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, g)
    names = g["names"]
    sets = {"all": names,
            "excl chowdary+singh": [n for n in names if not n.startswith(("chowdary", "singh"))],
            "excl chowdary+singh+nakayama": [n for n in names if not n.startswith(("chowdary", "singh", "nakayama"))]}
    T = pickle.load(open(f"results_thr{suf}.pkl", "rb"))
    for tag, sel in sets.items():
        with contextlib.redirect_stdout(io.StringIO()):
            out = g["run"](sel, tag)
        for v in ("smote_fs", "fs_smote"):
            out[f"H6 {v}"] = {m: float(np.mean([np.mean([T[n][(fs, v, c)][m] - T[n][(fs, "prior", c)][m] for n in sel])
                                                for fs in "FH" for c in ("knn", "lr")])) for m in ("gmean", "mcc")}
        key = f"{'2rep' if suf else '5rep'}|{tag}"
        res[key] = dict(n=len(sel), **out)
        print(f"\n== {key} | n={len(sel)}")
        for k in ("H1", "H2", "H3a", "H3b", "H4a", "H4b"):
            r = out[k]
            print(f"  {k:4s} {r['mean']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] {r['wins']}/{r['losses']} p={r['p']:.3f}"
                  + (f" holm={r['holm']:.3f}" if "holm" in r else ""))
        for v in ("smote_fs", "fs_smote"):
            print(f"  H6 {v}: G-mean {out['H6 ' + v]['gmean']:+.3f}  MCC {out['H6 ' + v]['mcc']:+.3f}")
L = pickle.load(open("results_leak2.pkl", "rb"))
null = defaultdict(list)
for n, k, r in L:
    if k >= 0:
        null[n].append(r)
for tag, ex in (("all", ()), ("excl chowdary+singh+nakayama", ("chowdary", "singh", "nakayama"))):
    ns = [n for n in null if not (ex and n.startswith(ex))]
    row = {f"{fs}|{c}|{p}": float(np.mean([np.mean([r[(fs, p, c)]["mcc"] for r in null[n]]) for n in ns]))
           for fs in "FH" for c in ("knn", "lr") for p in ("FS only", "SMOTE->FS", "FS->SMOTE", "leak SMOTE-first", "leak FS-first")}
    res[f"H5|{tag}"] = row
    inf = [v for k, v in row.items() if k.split("|")[2] in ("FS only", "SMOTE->FS", "FS->SMOTE")]
    sm = [v for k, v in row.items() if k.endswith("leak SMOTE-first")]
    print(f"\n== H5 | {tag} | n={len(ns)}: in-fold null MCC {min(inf):+.3f} to {max(inf):+.3f}; SMOTE-first {min(sm):+.3f} to {max(sm):+.3f}")
json.dump(res, open("summary_sens_ho.json", "w"), indent=1)
