"""MF-CSO (re-implemented) versus the Hellinger filter (same size and top-20) and the cap-20 wrapper, repeats 0-1.

Run from the repository root. Comparators come from the two-repeat results (results_cap_r2, results_xclf_r2), which use
the same outer folds. Writes hdi/summary_mfcso.json.
"""
import glob, json, os, pickle, re, numpy as np
from scipy.stats import wilcoxon
from fsimb.stats import nogueira_stability

ROOT = {"disc": "hdi", "held": "hdi_ho"}
load = lambda d: {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{d}/*.pkl"))}
M = {s: load(f"{r}/results_mfcso") for s, r in ROOT.items()}
C = {s: load(f"{r}/results_cap_r2") for s, r in ROOT.items()}
X = {s: load(f"{r}/results_xclf_r2") for s, r in ROOT.items()}
CL = {"knn": ("knn",), "lrsvm": ("lr", "svm"), "rf": ("rf",)}
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


def mval(s, n, m, cl):
    return float(np.mean([prim(M[s][n]["res"][(m, c)]) for c in CL[cl]]))


def oval(s, n, m, cl):  # m in {"cap20", "H20"} from the two-repeat wrapper experiment
    if cl == "knn":
        return prim(C[s][n]["res"][m])
    return float(np.mean([prim(X[s][n]["res"][(m, c)]) for c in CL[cl]]))


names = {s: sorted(set(M[s]) & set(C[s])) for s in M}
S = {"n": {s: len(v) for s, v in names.items()}, "mean": {}, "size": {}, "stab": {}, "time": {}, "contrasts": {}}
for s in M:
    S["mean"][s] = {f"{m}|{cl}": float(np.mean([mval(s, n, m, cl) for n in names[s]])) for m in ("mfcso", "H_same_k") for cl in CL}
    S["mean"][s].update({f"{m}|{cl}": float(np.mean([oval(s, n, m, cl) for n in names[s]])) for m in ("cap20", "H20") for cl in CL})
    S["mean"][s]["mfcso|1nn"] = float(np.mean([mval(s, n, "mfcso", "knn") * 0 + prim(M[s][n]["res"][("mfcso", "1nn")]) for n in names[s]]))
    S["size"][s] = dict(median=float(np.median([M[s][n]["res"][("mfcso", "size")] for n in names[s]])),
                        min=float(np.min([M[s][n]["res"][("mfcso", "size")] for n in names[s]])),
                        max=float(np.max([M[s][n]["res"][("mfcso", "size")] for n in names[s]])))
    S["stab"][s] = {m: float(np.mean([nogueira_stability(M[s][n]["subsets"][m], M[s][n]["p"]) for n in names[s]])) for m in ("mfcso", "H_same_k")}
    S["time"][s] = dict(median=float(np.median([np.median(M[s][n]["times"]) for n in names[s]])),
                        max=float(np.max([np.median(M[s][n]["times"]) for n in names[s]])))
CON = {"mfcso - H_same_k": lambda s, n, cl: mval(s, n, "mfcso", cl) - mval(s, n, "H_same_k", cl),
       "mfcso - H20": lambda s, n, cl: mval(s, n, "mfcso", cl) - oval(s, n, "H20", cl),
       "mfcso - cap20": lambda s, n, cl: mval(s, n, "mfcso", cl) - oval(s, n, "cap20", cl)}
for key, f in CON.items():
    for cl in CL:
        rows = [(s, n, f(s, n, cl)) for s in M for n in names[s]]
        out = {}
        for sets, tag in ((("disc",), "disc"), (("held",), "held"), (("disc", "held"), "both")):
            sel = [(s, n, x) for s, n, x in rows if s in sets]
            out[f"{tag}|dataset"] = summarize([x for *_, x in sel])
            fam = {}
            for s, n, x in sel: fam.setdefault((s, family(n)), []).append(x)
            out[f"{tag}|family"] = summarize([np.mean(x) for x in fam.values()])
        S["contrasts"][f"{key}|{cl}"] = out
json.dump(S, open("hdi/summary_mfcso.json", "w"), indent=1)
print("n", S["n"], "size", S["size"], "stab", S["stab"], "time", S["time"])
for s in M:
    print(s, {k: round(v, 3) for k, v in S["mean"][s].items()})
for k, d in S["contrasts"].items():
    print(f"{k:28s} " + "  ".join(f"{t}:{d[t]['mean']:+.3f}[{d[t]['ci90'][0]:+.3f},{d[t]['ci90'][1]:+.3f}] p={d[t]['p']:.3f} {d[t]['wins']}/{d[t]['losses']}{' E' if d[t]['equivalent'] else ''}"
                                  for t in ("disc|dataset", "held|dataset", "both|family")))
