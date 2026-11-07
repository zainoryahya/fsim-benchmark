"""Note: the Hellinger ranking here breaks ties with a stable sort; the wrapper experiments used numpy's default
sort, so H20 differs slightly between the two runs (ties are common with binned scores). Each contrast therefore uses
the Hellinger filter of its own run.

Additional baselines versus the Hellinger filter and the cap-20 wrapper (5 fold x 5 repeats).

Run from the repository root. Writes hdi/summary_base.json. Primary outcome = mean(G-mean, (MCC+1)/2, AUC-PR).
Contrasts are paired over datasets (and families), 90% bootstrap CI, equivalence within +-0.02.
"""
import glob, json, os, pickle, re, numpy as np
from scipy.stats import wilcoxon
from fsimb.stats import nogueira_stability

ROOT = {"disc": "hdi", "held": "hdi_ho"}
load = lambda d: {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{d}/*.pkl"))}
B = {s: load(f"{r}/results_base") for s, r in ROOT.items()}
C = {s: load(f"{r}/results_cap") for s, r in ROOT.items()}
X = {s: load(f"{r}/results_xclf") for s, r in ROOT.items()}
M = ["H", "F", "L1", "EN", "mRMR", "RFimp", "RFperm"]

CL = {"knn": ("knn",), "lrsvm": ("lr", "svm"), "rf": ("rf",)}
rng = np.random.default_rng(0)
prim = lambda r: float(np.mean([r[k] for k in ("gmean", "mcc", "aucpr")]))


def family(n):
    m = re.match(r"^(gisette|BASEHOCK|SMK_CAN_187|arcene|PCMAC|RELATHE|singh|chowdary)(_ir\d+)?$", n)
    return m.group(1) if m else n


def summarize(v, margin=0.02):
    v = np.asarray(v)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)])
    lo, hi = np.percentile(bs, [5, 95])
    p = float(wilcoxon(v).pvalue) if np.any(v != 0) else 1.0
    return dict(n=len(v), mean=float(v.mean()), ci90=[float(lo), float(hi)], p=p, wins=int((v > 0).sum()),
                losses=int((v < 0).sum()), equivalent=bool(lo > -margin and hi < margin))


def bval(s, n, m, k, cl):
    return float(np.mean([prim(B[s][n]["res"][(m, k, c)]) for c in CL[cl]]))


def wval(s, n, cl):            # cap-20 wrapper, same folds (5-NN from results_cap, others from results_xclf)
    if cl == "knn":
        return prim(C[s][n]["res"]["cap20"])
    return float(np.mean([prim(X[s][n]["res"][("cap20", c)]) for c in CL[cl]]))


names = {s: sorted(set(B[s]) & set(C[s])) for s in B}
S = {"n": {s: len(v) for s, v in names.items()}, "mean": {}, "vsH": {}, "vsWrap": {}, "stab": {}, "time": {}, "check": {}}
for s in B:
    S["mean"][s] = {f"{m}|{k}|{cl}": float(np.mean([bval(s, n, m, k, cl) for n in names[s]]))
                    for m in M for k in (10, 20, 30, 40) for cl in CL}
    S["mean"][s].update({f"wrap|20|{cl}": float(np.mean([wval(s, n, cl) for n in names[s]])) for cl in CL})
    S["stab"][s] = {m: float(np.mean([nogueira_stability(B[s][n]["subsets"][(m, 20)], B[s][n]["p"]) for n in names[s]])) for m in M}
    S["time"][s] = {m: float(np.median([np.median(B[s][n]["times"][m]) for n in names[s]])) for m in M}
    # consistency: H20 here vs H20 in the wrapper experiment (same folds), 5-NN
    S["check"][s] = float(np.max([abs(bval(s, n, "H", 20, "knn") - prim(C[s][n]["res"]["H20"])) for n in names[s]]))
for m in M[1:] + ["wrap"]:
    for cl in CL:
        for k in ((10, 20, 30, 40) if m != "wrap" else (20,)):
            key = f"{m}|{k}|{cl}"
            rows = []
            for s in B:
                for n in names[s]:
                    if m == "wrap":   # compare with the Hellinger top-20 of the same run (identical tie-breaking)
                        hw = prim(C[s][n]["res"]["H20"]) if cl == "knn" else float(np.mean([prim(X[s][n]["res"][("H20", c)]) for c in CL[cl]]))
                        rows.append((s, n, wval(s, n, cl) - hw))
                    else:
                        rows.append((s, n, bval(s, n, m, k, cl) - bval(s, n, "H", k, cl)))
            S["vsH"][key] = {}
            for sets, tag in ((("disc",), "disc"), (("held",), "held"), (("disc", "held"), "both")):
                sel = [(s, n, x) for s, n, x in rows if s in sets]
                S["vsH"][key][f"{tag}|dataset"] = summarize([x for *_, x in sel])
                fam = {}
                for s, n, x in sel: fam.setdefault((s, family(n)), []).append(x)
                S["vsH"][key][f"{tag}|family"] = summarize([np.mean(x) for x in fam.values()])
for m in M:
    for cl in CL:
        rows = [(s, n, bval(s, n, m, 20, cl) - wval(s, n, cl)) for s in B for n in names[s]]
        S["vsWrap"][f"{m}|20|{cl}"] = {tag: summarize([x for s, n, x in rows if s in sets])
                                      for sets, tag in ((("disc",), "disc"), (("held",), "held"), (("disc", "held"), "both"))}
json.dump(S, open("hdi/summary_base.json", "w"), indent=1)
print("n", S["n"], "H20 consistency (max abs diff)", S["check"])
for s in B:
    print(f"\n{s}: primary at k=20 (knn / lrsvm / rf), stability, median time per fold (s)")
    for m in M + ["wrap"]:
        row = " ".join(f"{S['mean'][s][f'{m}|20|{cl}']:.3f}" for cl in CL)
        extra = f" stab {S['stab'][s][m]:.2f} time {S['time'][s][m]:.2f}" if m != "wrap" else ""
        print(f"  {m:7s} {row}{extra}")
print("\nvs Hellinger at k=20, pooled families (mean [90% CI] E=equivalent)")
for m in M[1:] + ["wrap"]:
    print(f"  {m:7s} " + "  ".join(f"{cl}:{S['vsH'][f'{m}|20|{cl}']['both|family']['mean']:+.3f}"
                                   f"[{S['vsH'][f'{m}|20|{cl}']['both|family']['ci90'][0]:+.3f},{S['vsH'][f'{m}|20|{cl}']['both|family']['ci90'][1]:+.3f}]"
                                   f"{'E' if S['vsH'][f'{m}|20|{cl}']['both|family']['equivalent'] else ''}" for cl in CL))
