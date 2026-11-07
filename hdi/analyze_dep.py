"""Dependence-aware and equivalence analysis of the key contrasts (discovery + held-out pooled).

Units: datasets, or families (controlled-IR variants of one base set count once; each synthetic set and
each other set is its own family). For each contrast: mean difference over units, 90% bootstrap CI over
units, Wilcoxon p, and equivalence (TOST-style) when the 90% CI lies inside +-0.02.
"""
import glob, os, pickle, re, json, numpy as np
from scipy.stats import wilcoxon

def load(root, d): return {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{root}/{d}/*.pkl"))}
ROOTS = {"disc": "hdi", "held": "hdi_ho"}
D = {k: {d: load(r, d) for d in ("results_wrapH", "results_cap", "results_emt", "results_xclf", "results_ridge")} for k, r in ROOTS.items()}
THR = {k: pickle.load(open(f"{r}/results_thr.pkl", "rb")) for k, r in ROOTS.items()}
CEIL = {"chiaretti", "CLL_SUB_111", "lymphoma", "Carcinom"}

def family(n):
    m = re.match(r"^(gisette|BASEHOCK|SMK_CAN_187|arcene|PCMAC|RELATHE|singh|chowdary)(_ir\d+)?$", n)
    return m.group(1) if m else n

def prim(res): return np.mean([res[k] for k in ("gmean", "mcc", "aucpr")])
def clf_prim(R, n, m, cl): return np.mean([R[n]["res"][(m, cl)][k] for k in ("gmean", "mcc", "aucpr")])

contrasts = {
 "cap20 - H20 (5-NN final)": lambda s, n: prim(D[s]["results_cap"][n]["res"]["cap20"]) - prim(D[s]["results_cap"][n]["res"]["H20"]),
 "cap20 - H20 (LR/SVM final)": lambda s, n: np.mean([clf_prim(D[s]["results_xclf"], n, "cap20", c) - clf_prim(D[s]["results_xclf"], n, "H20", c) for c in ("lr", "svm")]),
 "ridge_cap20 - H20 (ridge/LR/SVM)": lambda s, n: np.mean([clf_prim(D[s]["results_ridge"], n, "ridge_cap20", c) - clf_prim(D[s]["results_ridge"], n, "H20", c) for c in ("rls", "lr", "svm")]),
 "emt_min3 - cap20 (5-NN)": lambda s, n: prim(D[s]["results_emt"][n]["res"]["emt_min3"]) - prim(D[s]["results_cap"][n]["res"]["cap20"]),
 "bbrier - acc wrapper (5-NN, H pool)": lambda s, n: prim(D[s]["results_wrapH"][n]["res"]["bbrier_a99"]) - prim(D[s]["results_wrapH"][n]["res"]["acc_a99"]),
 "SMOTE->FS - prior, G-mean": lambda s, n: np.mean([THR[s][n][(fs, "smote_fs", c)]["gmean"] - THR[s][n][(fs, "prior", c)]["gmean"] for fs in "FH" for c in ("knn", "lr")]),
 "SMOTE->FS - prior, MCC": lambda s, n: np.mean([THR[s][n][(fs, "smote_fs", c)]["mcc"] - THR[s][n][(fs, "prior", c)]["mcc"] for fs in "FH" for c in ("knn", "lr")]),
 "FS->SMOTE - prior, G-mean": lambda s, n: np.mean([THR[s][n][(fs, "fs_smote", c)]["gmean"] - THR[s][n][(fs, "prior", c)]["gmean"] for fs in "FH" for c in ("knn", "lr")]),
 "FS->SMOTE - prior, MCC": lambda s, n: np.mean([THR[s][n][(fs, "fs_smote", c)]["mcc"] - THR[s][n][(fs, "prior", c)]["mcc"] for fs in "FH" for c in ("knn", "lr")]),
}
rng = np.random.default_rng(1)
def summarize(v, margin=0.02):
    v = np.asarray(v)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [5, 95])
    p = float(wilcoxon(v).pvalue) if np.any(v != 0) else 1.0
    return dict(n=len(v), mean=float(v.mean()), ci90=[float(lo), float(hi)], p=p, equivalent=bool(lo > -margin and hi < margin))
out = {}
print(f"{'contrast':38s} {'set':6s} {'units':8s} {'n':>3s} {'mean':>8s} {'90% CI':>18s} {'p':>6s} equiv(+-0.02)")
for name, f in contrasts.items():
    rows = []
    for s in ("disc", "held"):
        for n in sorted(D[s]["results_cap"]):
            if n in CEIL or n not in THR[s]: continue
            try: rows.append((s, n, f(s, n)))
            except KeyError: pass
    out[name] = {}
    for sets, tag in ((("disc",), "disc"), (("held",), "held"), (("disc", "held"), "both")):
        sel = [(s, n, v) for s, n, v in rows if s in sets]
        for unit in ("dataset", "family"):
            if unit == "dataset": vals = [v for _, _, v in sel]
            else:
                fam = {}
                for s, n, v in sel: fam.setdefault((s, family(n)), []).append(v)
                vals = [np.mean(x) for x in fam.values()]
            r = summarize(vals); out[name][f"{tag}|{unit}"] = r
            print(f"{name:38s} {tag:6s} {unit:8s} {r['n']:3d} {r['mean']:+8.4f} ({r['ci90'][0]:+.3f},{r['ci90'][1]:+.3f}) {r['p']:6.3f} {r['equivalent']}")
json.dump(out, open("hdi/summary_dep.json", "w"), indent=1)
