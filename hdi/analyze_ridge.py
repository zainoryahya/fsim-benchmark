import glob, os, pickle, json, csv, numpy as np
from scipy.stats import wilcoxon
from fsimb.stats import nogueira_stability
from run_ridge import NEW, SAVED, CLFS
R = {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob("results_ridge/*.pkl"))}
names = sorted(R)
IR = {r["name"]: float(r["IR"]) for r in csv.DictReader(open("data/v2/inventory.csv"))}
M = NEW + list(SAVED) + ["F20"]
ix = {m: i for i, m in enumerate(M)}
def prim(n, m, c): return np.mean([R[n]["res"][(m, c)][k] for k in ("gmean", "mcc", "aucpr")])
P = {c: np.array([[prim(n, m, c) for m in M] for n in names]) for c in CLFS}
P["avg"] = np.mean([P[c] for c in CLFS], axis=0)
SZ = np.array([[R[n]["res"][(m, "size")] for m in M] for n in names])
ST = np.array([[nogueira_stability(R[n]["subsets"][m], R[n]["p"]) for m in M] for n in names])
def holm(ps):
    o = np.argsort(ps); k = len(ps); out = np.zeros(k); run = 0.0
    for j, i in enumerate(o):
        run = max(run, min(1.0, (k - j) * ps[i])); out[i] = run
    return out
def pair(a, b, A):
    d = A[:, ix[a]] - A[:, ix[b]]
    return dict(mean=float(d.mean()), median=float(np.median(d)), wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
tests = [("ridge_cap10", "H10"), ("ridge_cap20", "H20"), ("ridge_cap30", "H30"), ("ridge_free", "H40"),
         ("ridge_cap20", "ridge_free"), ("ridge_cap20", "cap20_knn"), ("ridge_cap20", "F20"), ("ridge_cap10", "ridge_cap20"),
         ("ridge_cap30", "ridge_cap20")]
S = {"n": len(names), "mean": {c: dict(zip(M, P[c].mean(0).round(4).tolist())) for c in P},
     "size": dict(zip(M, SZ.mean(0).round(1).tolist())), "stab": dict(zip(M, ST.mean(0).round(3).tolist()))}
S["tests"] = {c: {f"{a} vs {b}": pair(a, b, P[c]) for a, b in tests} for c in P}
for c in P:
    keys = ["ridge_cap10 vs H10", "ridge_cap20 vs H20", "ridge_cap30 vs H30"]
    adj = holm(np.array([S["tests"][c][k]["p"] for k in keys]))
    for k, a in zip(keys, adj): S["tests"][c][k]["holm"] = float(a)
g = lambda n: "IR<3" if IR[n] < 3 else "IR3-8" if IR[n] <= 8 else "IR>8"
S["by_IR_avg"] = {k: dict(n=sum(g(n) == k for n in names),
                          **{m: float(P["avg"][[i for i, n in enumerate(names) if g(n) == k], ix[m]].mean()) for m in M}) for k in ("IR<3", "IR3-8", "IR>8")}
json.dump(S, open("summary_ridge.json", "w"), indent=1)
print(f"n={len(names)}\n{'method':11s}" + "".join(f"{c:>7s}" for c in P) + "   size  stab")
for m in M: print(f"{m:11s}" + "".join(f"{P[c][:, ix[m]].mean():7.3f}" for c in P) + f"  {SZ[:, ix[m]].mean():5.1f} {ST[:, ix[m]].mean():.3f}")
for c in P:
    print("\n==", c)
    for k, v in S["tests"][c].items():
        print(f"  {k:28s} {v['mean']:+.4f} med {v['median']:+.4f} {v['wins']:2d}/{v['losses']:2d} p={v['p']:.3f}" + (f" holm={v['holm']:.3f}" if "holm" in v else ""))
for k, v in S["by_IR_avg"].items(): print(k, v["n"], {m: round(v[m], 3) for m in ("ridge_free", "ridge_cap20", "H20", "cap20_knn")})
