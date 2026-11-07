"""sssHD (re-implemented) versus the Hellinger filter of the same run (same folds, repeats 0-1).

Run from the repository root. Writes hdi/summary_ssshd.json. Primary outcome = mean(G-mean, (MCC+1)/2, AUC-PR).
Contrasts paired over datasets and over families (unweighted mean of a family's tasks), 90% bootstrap CI,
two-sided Wilcoxon, equivalence within +-0.02.
"""
import glob, json, os, pickle, re, numpy as np
from scipy.stats import wilcoxon
from fsimb.stats import nogueira_stability

ROOT = {"disc": "hdi", "held": "hdi_ho"}
load = lambda d: {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{d}/*.pkl"))}
R = {s: load(f"{r}/results_ssshd") for s, r in ROOT.items()}
CL = {"knn": ("knn",), "lrsvm": ("lr", "svm"), "rf": ("rf",)}
KS = (10, 20, 30, 40)
prim = lambda r: float(np.mean([r[k] for k in ("gmean", "mcc", "aucpr")]))
rng = np.random.default_rng(0)


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


def val(s, n, m, k, cl):
    return float(np.mean([prim(R[s][n]["res"][(m, k, c)]) for c in CL[cl]]))


S = {"n": {s: len(v) for s, v in R.items()}, "mean": {}, "vsH": {}, "stab": {}, "time": {}, "sel": {}}
for s in R:
    names = sorted(R[s])
    S["mean"][s] = {f"{m}|{k}|{cl}": float(np.mean([val(s, n, m, k, cl) for n in names])) for m in ("sssHD", "H") for k in KS for cl in CL}
    S["stab"][s] = {m: float(np.mean([nogueira_stability(R[s][n]["subsets"][(m, 20)], R[s][n]["p"]) for n in names])) for m in ("sssHD", "H")}
    S["time"][s] = {m: dict(median=float(np.median([np.median(R[s][n]["times"][m]) for n in names])),
                            max=float(np.max([np.median(R[s][n]["times"][m]) for n in names]))) for m in ("sssHD", "H")}
    nsel = [np.median([i["n_selected"] for i in R[s][n]["info"]]) for n in names]
    sub = [np.median([np.median(i["sizes"]) for i in R[s][n]["info"]]) for n in names]
    S["sel"][s] = dict(n_selected_median=float(np.median(nsel)), n_selected_min=float(np.min(nsel)),
                       share_below_40=float(np.mean([np.mean([i["n_selected"] < 40 for i in R[s][n]["info"]]) for n in names])),
                       subsample_size_median=float(np.median(sub)))
for k in KS:
    for cl in CL:
        rows = [(s, n, val(s, n, "sssHD", k, cl) - val(s, n, "H", k, cl)) for s in R for n in R[s]]
        out = {}
        for sets, tag in ((("disc",), "disc"), (("held",), "held"), (("disc", "held"), "both")):
            sel = [(s, n, x) for s, n, x in rows if s in sets]
            out[f"{tag}|dataset"] = summarize([x for *_, x in sel])
            fam = {}
            for s, n, x in sel: fam.setdefault((s, family(n)), []).append(x)
            out[f"{tag}|family"] = summarize([np.mean(x) for x in fam.values()])
        S["vsH"][f"{k}|{cl}"] = out
json.dump(S, open("hdi/summary_ssshd.json", "w"), indent=1)
print("n", S["n"], "\nstab", S["stab"], "\ntime", S["time"], "\nsel", S["sel"])
for s in R:
    print(s, {k: round(v, 3) for k, v in S["mean"][s].items() if "|20|" in k})
for key, d in S["vsH"].items():
    print(f"{key:9s} " + "  ".join(f"{t}:{d[t]['mean']:+.3f}[{d[t]['ci90'][0]:+.3f},{d[t]['ci90'][1]:+.3f}] p={d[t]['p']:.3f} {d[t]['wins']}/{d[t]['losses']}{' E' if d[t]['equivalent'] else ''}"
                               for t in ("disc|dataset", "held|dataset", "both|dataset", "both|family")))
