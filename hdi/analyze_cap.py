import glob, os, pickle, json, csv, numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
from fsimb.stats import nogueira_stability, bayes_signed_rank
from run_cap import METHODS, CAPS
R = {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob("results_cap/*.pkl"))}
H = {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob("results_wrapH/*.pkl"))}
names = sorted(R)
IR = {r["name"]: float(r["IR"]) for r in csv.DictReader(open("data/v2/inventory.csv"))}
cols = ["single"] + METHODS
ix = {m: i for i, m in enumerate(cols)}
def res(n, m): return H[n]["res"]["bbrier_a99"] if m == "single" else R[n]["res"][m]
def sub(n, m): return H[n]["subsets"]["bbrier_a99"] if m == "single" else R[n]["subsets"][m]
P = np.array([[np.mean([res(n, m)[k] for k in ("gmean", "mcc", "aucpr")]) for m in cols] for n in names])
G = np.array([[res(n, m)["gmean"] for m in cols] for n in names])
MC = np.array([[res(n, m)["mcc"] for m in cols] for n in names])
AP = np.array([[res(n, m)["aucpr"] for m in cols] for n in names])
SZ = np.array([[res(n, m)["size"] for m in cols] for n in names])
ST = np.array([[nogueira_stability(sub(n, m), R[n]["p"]) for m in cols] for n in names])
def holm(ps):
    o = np.argsort(ps); k = len(ps); out = np.zeros(k); run = 0.0
    for j, i in enumerate(o):
        run = max(run, min(1.0, (k - j) * ps[i])); out[i] = run
    return out
def pair(a, b, A=P):
    d = A[:, ix[a]] - A[:, ix[b]]
    return dict(mean=float(d.mean()), median=float(np.median(d)), wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
S = {"n": len(names)}
for k, A in (("primary", P), ("gmean", G), ("mcc", MC), ("aucpr", AP), ("size", SZ), ("stab", ST)):
    S[k] = dict(zip(cols, A.mean(0).tolist()))
S["cap_vs_filter"] = {f"cap{c} vs H{c}": pair(f"cap{c}", f"H{c}") for c in CAPS}
adj = holm(np.array([v["p"] for v in S["cap_vs_filter"].values()]))
for v, a in zip(S["cap_vs_filter"].values(), adj): v["holm"] = float(a)
S["vs_single"] = {m: pair(m, "single") for m in METHODS if m.startswith(("cap", "auto"))}
S["contrasts"] = {f"{a} vs {b}": pair(a, b) for a, b in (("auto_loo", "cap20"), ("auto_cv", "cap20"), ("auto_cv", "auto_loo"),
        ("cap10", "cap20"), ("cap30", "cap20"), ("cap40", "cap20"), ("auto_cv", "H20"))}
S["bayes"] = {f"{a} vs {b}": dict(zip(["P(A<B)", "P(rope)", "P(A>B)"], bayes_signed_rank(P[:, ix[a]], P[:, ix[b]])))
              for a, b in (("cap20", "H20"), ("auto_cv", "cap20"), ("auto_loo", "cap20"))}
st, p = friedmanchisquare(*[P[:, ix[f"cap{c}"]] for c in CAPS]); S["friedman_caps"] = dict(stat=float(st), p=float(p))
S["chosen_cap_share"] = {a: {c: float(np.mean([x == c for n in names for x in R[n]["chosen"][a]])) for c in CAPS}
                         for a in ("auto_loo", "auto_cv")}
g = lambda n: "IR<3" if IR[n] < 3 else "IR3-8" if IR[n] <= 8 else "IR>8"
S["by_IR"] = {k: dict(n=sum(g(n) == k for n in names), **{m: float(P[[i for i, n in enumerate(names) if g(n) == k], ix[m]].mean()) for m in cols})
              for k in ("IR<3", "IR3-8", "IR>8")}
json.dump(S, open("summary_cap.json", "w"), indent=1)
print(f"{'method':9s} primary gmean   mcc  aucpr   size  stab")
for m in cols: i = ix[m]; print(f"{m:9s} {P[:,i].mean():.3f}  {G[:,i].mean():.3f} {MC[:,i].mean():.3f} {AP[:,i].mean():.3f} {SZ[:,i].mean():6.1f} {ST[:,i].mean():.3f}")
for k, v in {**S["cap_vs_filter"], **S["vs_single"], **S["contrasts"]}.items():
    print(f"{k:22s} {v['mean']:+.4f} med {v['median']:+.4f} {v['wins']:2d}/{v['losses']:2d} p={v['p']:.3f}" + (f" holm={v['holm']:.3f}" if "holm" in v else ""))
print("friedman caps", S["friedman_caps"]); print("bayes", json.dumps(S["bayes"])); print("chosen", S["chosen_cap_share"])
for k, v in S["by_IR"].items(): print(k, v["n"], {m: round(v[m], 3) for m in cols})
