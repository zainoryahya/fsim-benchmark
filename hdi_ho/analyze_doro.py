"""Dorothea additional validation: descriptive results for predictions P1-P5 (dorothea_prereg.md)."""
import os, pickle, json, numpy as np
from fsimb.stats import nogueira_stability
L = lambda d: pickle.load(open(f"{d}/dorothea.pkl", "rb")) if os.path.exists(f"{d}/dorothea.pkl") else None
W, WH, C, E, X, R = (L(d) for d in ("results_doro_wrap", "results_doro_wrapH", "results_doro_cap", "results_doro_emt", "results_doro_xclf", "results_doro_ridge"))
prim = lambda r: float(np.mean([r[k] for k in ("gmean", "mcc", "aucpr")]))
S = {}
p = C["p"]
S["primary_5nn"] = {**{m: prim(C["res"][m]) for m in C["res"]}, **{m: prim(E["res"][m]) for m in E["res"]},
                    "single_bbrier": prim(WH["res"]["bbrier_a99"]), "single_acc": prim(WH["res"]["acc_a99"])}
S["size"] = {m: C["res"][m]["size"] for m in C["res"]}
S["stability"] = {**{m: nogueira_stability(C["subsets"][m], p) for m in C["subsets"]},
                  **{m: nogueira_stability(E["subsets"][m], p) for m in E["subsets"]},
                  "single_bbrier": nogueira_stability(WH["subsets"]["bbrier_a99"], p)}
S["P4_emt_min3_minus_cap20"] = S["primary_5nn"]["emt_min3"] - S["primary_5nn"]["cap20"]
S["cap_minus_H_5nn"] = {k: S["primary_5nn"][f"cap{k}"] - S["primary_5nn"][f"H{k}"] for k in (10, 20, 30, 40)}
S["raw_mcc_H20"] = 2 * C["res"]["H20"]["mcc"] - 1
if X:
    xp = lambda m, cls: float(np.mean([np.mean([X["res"][(m, c)][k] for k in ("gmean", "mcc", "aucpr")]) for c in cls]))
    S["P3_cap20_minus_H20_lrsvm"] = xp("cap20", ("lr", "svm")) - xp("H20", ("lr", "svm"))
    S["cap_minus_H_lrsvm"] = {k: xp(f"cap{k}", ("lr", "svm")) - xp(f"H{k}", ("lr", "svm")) for k in (10, 20, 30, 40)}
    S["emt_min3_minus_H20_lrsvm"] = xp("emt_min3", ("lr", "svm")) - xp("H20", ("lr", "svm"))
    S["rf_cap20_minus_H20"] = xp("cap20", ("rf",)) - xp("H20", ("rf",))
if R:
    rp = lambda m, cls: float(np.mean([np.mean([R["res"][(m, c)][k] for k in ("gmean", "mcc", "aucpr")]) for c in cls]))
    S["ridge_cap20_minus_H20_avg"] = rp("ridge_cap20", ("rls", "lr", "svm")) - rp("H20", ("rls", "lr", "svm"))
for f, key in (("results_doro_leak2.pkl", "leak"), ("results_doro_thr.pkl", "thr")):
    if os.path.exists(f):
        S[key] = "present"
if os.path.exists("results_doro_thr.pkl"):
    T = pickle.load(open("results_doro_thr.pkl", "rb"))["dorothea"]
    S["thr"] = {v: {m: float(np.mean([T[(fs, v, c)][m] for fs in "FH" for c in ("knn", "lr")])) for m in ("gmean", "mcc")}
                for v in ("base", "prior", "tuned", "smote_fs", "fs_smote")}
    S["P2_smote_fs_minus_prior_gmean"] = S["thr"]["smote_fs"]["gmean"] - S["thr"]["prior"]["gmean"]
    S["P2_fs_smote_minus_prior_gmean"] = S["thr"]["fs_smote"]["gmean"] - S["thr"]["prior"]["gmean"]
if os.path.exists("results_doro_leak2.pkl"):
    out = pickle.load(open("results_doro_leak2.pkl", "rb"))
    real = [r for n, k, r in out if k < 0][0]; null = [r for n, k, r in out if k >= 0]
    S["leak"] = {}
    for pipe in ("FS only", "SMOTE->FS", "FS->SMOTE", "leak SMOTE-first", "leak FS-first"):
        S["leak"][pipe] = {f"{c[0]}|{c[1]}": dict(null_mcc=float(np.mean([r[c + (pipe,)][0] if False else r[(c[0], pipe, c[1])]["mcc"] for r in null])),
                                   null_auc=float(np.mean([r[(c[0], pipe, c[1])]["auc"] for r in null])),
                                   real_mcc=real[(c[0], pipe, c[1])]["mcc"]) for c in (("F", "knn"), ("F", "lr"), ("H", "knn"), ("H", "lr"))}
if os.path.exists("results_doro_base/dorothea.pkl"):     # further baselines (run_base.py), merged by merge_doro.py
    B = L("results_doro_base")
    GR = {"knn": ("knn",), "lrsvm": ("lr", "svm"), "rf": ("rf",)}
    bp = lambda m, k, cls: float(np.mean([prim(B["res"][(m, k, c)]) for c in cls]))
    meths = sorted({m for m, _, _ in B["res"]} - {"H"})
    S["base_minus_H"] = {f"{m}|{k}|{g}": bp(m, k, cls) - bp("H", k, cls) for m in meths for k in (10, 20, 30, 40)
                         for g, cls in GR.items()}
    S["base_time_median_s"] = {m: float(np.median(t)) for m, t in B["times"].items()}
json.dump(S, open("summary_doro.json", "w"), indent=1, default=str)
print(json.dumps(S, indent=1, default=str))
