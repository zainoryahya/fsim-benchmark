import glob, os, pickle, json, csv, numpy as np
from scipy.stats import wilcoxon
from fsimb.stats import nogueira_stability, bayes_signed_rank
load = lambda d: {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{d}/*.pkl"))}
C, E, H, F = load("results_ctrl"), load("results_emt"), load("results_wrapH"), load("results_wrap")
names = sorted(C)
IR = {r["name"]: float(r["IR"]) for r in csv.DictReader(open("data/v2/inventory.csv"))}
def res(n, m):
    if m == "single": return H[n]["res"]["bbrier_a99"]
    if m == "H-top20": return F[n]["res"]["H-top20"]
    if m.startswith("single_"): return C[n]["res"][m]
    return E[n]["res"][m]
def sub(n, m):
    if m == "single": return H[n]["subsets"]["bbrier_a99"]
    if m == "H-top20": return F[n]["subsets"]["H-top20"]
    if m.startswith("single_"): return C[n]["subsets"][m]
    return E[n]["subsets"][m]
cols = ["single", "single_top50", "single_cap20", "emt_f", "emt_relief", "emt_min3", "emt_min3_rl", "H-top20"]
ix = {m: i for i, m in enumerate(cols)}
P = np.array([[np.mean([res(n, m)[k] for k in ("gmean", "mcc", "aucpr")]) for m in cols] for n in names])
SZ = np.array([[res(n, m).get("size", 20) for m in cols] for n in names])
ST = np.array([[nogueira_stability(sub(n, m), C[n]["p"]) for m in cols] for n in names])
def pair(a, b):
    d = P[:, ix[a]] - P[:, ix[b]]
    return dict(mean=float(d.mean()), median=float(np.median(d)), wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
S = dict(mean=dict(zip(cols, P.mean(0).round(4).tolist())), size=dict(zip(cols, SZ.mean(0).round(1).tolist())),
         stab=dict(zip(cols, ST.mean(0).round(3).tolist())))
pairs = [("single_top50", "single"), ("single_cap20", "single"), ("emt_min3", "single_top50"), ("emt_min3", "single_cap20"),
         ("emt_f", "single_top50"), ("emt_f", "single_cap20"), ("single_top50", "H-top20"), ("single_cap20", "H-top20"),
         ("single_cap20", "single_top50")]
S["pairs"] = {f"{a} vs {b}": pair(a, b) for a, b in pairs}
S["bayes"] = {f"{a} vs {b}": dict(zip(["P(A<B)", "P(rope)", "P(A>B)"], bayes_signed_rank(P[:, ix[a]], P[:, ix[b]])))
              for a, b in (("emt_min3", "single_top50"), ("emt_min3", "single_cap20"))}
g = lambda n: "IR<3" if IR[n] < 3 else "IR3-8" if IR[n] <= 8 else "IR>8"
S["by_IR"] = {k: dict(n=len([n for n in names if g(n) == k]),
                      **{m: float(P[[i for i, n in enumerate(names) if g(n) == k], ix[m]].mean()) for m in cols})
              for k in ("IR<3", "IR3-8", "IR>8")}
json.dump(S, open("summary_ctrl.json", "w"), indent=1)
print(f"{'method':14s} primary  size  stab")
for m in cols: print(f"{m:14s} {P[:,ix[m]].mean():.3f} {SZ[:,ix[m]].mean():6.1f} {ST[:,ix[m]].mean():.3f}")
for k, v in S["pairs"].items(): print(f"{k:34s} {v['mean']:+.4f} med {v['median']:+.4f} {v['wins']:2d}/{v['losses']:2d} p={v['p']:.3f}")
print(json.dumps(S["bayes"]))
for k, v in S["by_IR"].items(): print(k, v["n"], {m: round(v[m], 3) for m in cols})
