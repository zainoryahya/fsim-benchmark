import glob, os, pickle, json, csv, numpy as np
from scipy.stats import wilcoxon
from run_xclf import SRC, CLFS
R = {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob("results_xclf/*.pkl"))}
names = sorted(R)
IR = {r["name"]: float(r["IR"]) for r in csv.DictReader(open("data/v2/inventory.csv"))}
M = list(SRC)
def prim(n, m, c): return np.mean([R[n]["res"][(m, c)][k] for k in ("gmean", "mcc", "aucpr")])
P = {c: np.array([[prim(n, m, c) for m in M] for n in names]) for c in CLFS}
P["avg"] = np.mean([P[c] for c in CLFS], axis=0)
ix = {m: i for i, m in enumerate(M)}
def holm(ps):
    o = np.argsort(ps); k = len(ps); out = np.zeros(k); run = 0.0
    for j, i in enumerate(o):
        run = max(run, min(1.0, (k - j) * ps[i])); out[i] = run
    return out
def pair(a, b, A):
    d = A[:, ix[a]] - A[:, ix[b]]
    return dict(mean=float(d.mean()), median=float(np.median(d)), wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
S = {"n": len(names), "mean": {c: dict(zip(M, P[c].mean(0).round(4).tolist())) for c in P}}
tests = [("cap10", "H10"), ("cap20", "H20"), ("cap30", "H30"), ("cap40", "H40"), ("cap20", "single"), ("auto_cv", "H20"),
         ("auto_cv", "cap20"), ("emt_min3", "cap20"), ("emt_min3", "single"), ("cap20", "F20")]
S["tests"] = {c: {f"{a} vs {b}": pair(a, b, P[c]) for a, b in tests} for c in P}
for c in P:   # Holm over the four cap-vs-filter tests, within classifier
    keys = [f"cap{k} vs H{k}" for k in (10, 20, 30, 40)]
    adj = holm(np.array([S["tests"][c][k]["p"] for k in keys]))
    for k, a in zip(keys, adj): S["tests"][c][k]["holm"] = float(a)
g = lambda n: "IR<3" if IR[n] < 3 else "IR3-8" if IR[n] <= 8 else "IR>8"
S["by_IR_avg"] = {k: dict(n=sum(g(n) == k for n in names),
                          **{m: float(P["avg"][[i for i, n in enumerate(names) if g(n) == k], ix[m]].mean()) for m in M})
                  for k in ("IR<3", "IR3-8", "IR>8")}
json.dump(S, open("summary_xclf.json", "w"), indent=1)
show = ["single", "F20", "emt_min3", "cap10", "cap20", "cap30", "cap40", "auto_loo", "auto_cv", "H10", "H20", "H30", "H40"]
print(f"{'method':10s}" + "".join(f"{c:>7s}" for c in P))
for m in show: print(f"{m:10s}" + "".join(f"{P[c][:, ix[m]].mean():7.3f}" for c in P))
for c in P:
    print("\n==", c)
    for k, v in S["tests"][c].items():
        print(f"  {k:22s} {v['mean']:+.4f} med {v['median']:+.4f} {v['wins']:2d}/{v['losses']:2d} p={v['p']:.3f}" + (f" holm={v['holm']:.3f}" if "holm" in v else ""))
for k, v in S["by_IR_avg"].items(): print(k, v["n"], {m: round(v[m], 3) for m in ("single", "cap20", "auto_cv", "H20", "emt_min3")})
