"""Phase 3 analysis -> summary_emt.json and printed tables.

Primary outcome (as in Phase 2): per dataset, mean of outer-test G-mean,
normalised MCC and AUC-PR. Baseline: single-task BGWO with the same fitness and
Hellinger pool (bbrier_a99 in results_wrapH).
"""
import csv, glob, json, os, pickle
import numpy as np
from scipy.stats import wilcoxon
from fsimb.stats import nogueira_stability, bayes_signed_rank
from run_emt import METHODS

load = lambda d: {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{d}/*.pkl"))}
E, H, F = load("results_emt"), load("results_wrapH"), load("results_wrap")
names = sorted(E)
IR = {r["name"]: float(r["IR"]) for r in csv.DictReader(open("data/v2/inventory.csv"))}

cols = ["single"] + METHODS + ["H-top20"]
def get(n, m, key):
    if m == "single":
        return H[n]["res"]["bbrier_a99"][key]
    if m == "H-top20":
        return F[n]["res"]["H-top20"][key]
    return E[n]["res"][m][key]
def sub(n, m):
    if m == "single":
        return H[n]["subsets"]["bbrier_a99"]
    if m == "H-top20":
        return F[n]["subsets"]["H-top20"]
    return E[n]["subsets"][m]

P = np.array([[np.mean([get(n, m, k) for k in ("gmean", "mcc", "aucpr")]) for m in cols] for n in names])
G = np.array([[get(n, m, "gmean") for m in cols] for n in names])
MC = np.array([[get(n, m, "mcc") for m in cols] for n in names])
AP = np.array([[get(n, m, "aucpr") for m in cols] for n in names])
SZ = np.array([[get(n, m, "size") if m != "H-top20" else 20 for m in cols] for n in names])
ST = np.array([[nogueira_stability(sub(n, m), E[n]["p"]) for m in cols] for n in names])
ix = {m: i for i, m in enumerate(cols)}


def holm(ps):
    o = np.argsort(ps); k = len(ps); out = np.zeros(k); run = 0.0
    for j, i in enumerate(o):
        run = max(run, min(1.0, (k - j) * ps[i])); out[i] = run
    return out


def paired(a, b, A=P):
    d = A[:, ix[a]] - A[:, ix[b]]
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return dict(mean=float(d.mean()), median=float(np.median(d)), wins=int((d > 0).sum()),
                losses=int((d < 0).sum()), p=p)


S = {"n_datasets": len(names), "methods": cols}
for key, A in (("primary", P), ("gmean", G), ("mcc", MC), ("aucpr", AP), ("size", SZ), ("stability", ST)):
    S[f"mean_{key}"] = dict(zip(cols, A.mean(0).tolist()))
S["vs_single"] = {m: paired(m, "single") for m in METHODS}
adj = holm(np.array([S["vs_single"][m]["p"] for m in METHODS]))
for m, a in zip(METHODS, adj):
    S["vs_single"][m]["holm"] = float(a)
S["vs_hfilter"] = {m: paired(m, "H-top20") for m in ["single"] + METHODS}
S["key_contrasts"] = {f"{a} vs {b}": paired(a, b) for a, b in (
    ("emt_min3", "emt_rand3"), ("emt_min3", "emt_relief"), ("emt_min3", "emt_f"),
    ("emt_min3_rl", "emt_min3"), ("emt_min3_rl", "emt_rand3"))}
S["bayes"] = {f"{a} vs {b}": dict(zip(["P(A<B)", "P(rope)", "P(A>B)"], bayes_signed_rank(P[:, ix[a]], P[:, ix[b]])))
              for a, b in (("emt_min3", "single"), ("emt_min3_rl", "single"), ("emt_min3", "emt_rand3"),
                           ("emt_min3_rl", "emt_min3"), ("emt_min3_rl", "H-top20"))}

def grp(n):
    return "IR<3" if IR[n] < 3 else "IR3-8" if IR[n] <= 8 else "IR>8"
S["by_IR"] = {}
for g in sorted({grp(n) for n in names}):
    sel = [i for i, n in enumerate(names) if grp(n) == g]
    S["by_IR"][g] = dict(n=len(sel), **{m: float(P[sel, ix[m]].mean()) for m in cols})

# bandit behaviour and which task wins
picks = np.zeros(3); win = np.zeros(4); q = []
for n in names:
    for inf in E[n]["infos"]["emt_min3_rl"]:
        picks += np.array(inf["picks"]) / sum(inf["picks"])
        q.append(inf["q"])
    for inf in E[n]["infos"]["emt_min3"]:
        win[inf["winner_task"]] += 1
nrun = sum(len(E[n]["infos"]["emt_min3_rl"]) for n in names)
S["bandit_pick_share_(hell,auc,bfisher)"] = (picks / nrun).tolist()
S["bandit_mean_q"] = np.mean(q, axis=0).tolist()
S["emt_min3_winner_share_(main,hell,auc,bfisher)"] = (win / win.sum()).tolist()
S["per_dataset"] = {n: {m: float(P[i, ix[m]]) for m in ("single", "emt_min3", "emt_min3_rl", "emt_rand3", "H-top20")}
                    for i, n in enumerate(names)}
json.dump(S, open("summary_emt.json", "w"), indent=1)

print("datasets:", len(names))
print(f"\n{'method':12s} {'primary':>8s} {'gmean':>7s} {'mcc':>7s} {'aucpr':>7s} {'size':>6s} {'stab':>6s}")
for m in cols:
    i = ix[m]
    print(f"{m:12s} {P[:,i].mean():8.3f} {G[:,i].mean():7.3f} {MC[:,i].mean():7.3f} {AP[:,i].mean():7.3f} {SZ[:,i].mean():6.1f} {ST[:,i].mean():6.3f}")
print("\nvs single (mean, median, W/L, p, holm)")
for m, v in S["vs_single"].items():
    print(f"  {m:12s} {v['mean']:+.4f} {v['median']:+.4f} {v['wins']:2d}/{v['losses']:2d} p={v['p']:.3f} holm={v['holm']:.3f}")
print("\nvs H-top20")
for m, v in S["vs_hfilter"].items():
    print(f"  {m:12s} {v['mean']:+.4f} {v['median']:+.4f} {v['wins']:2d}/{v['losses']:2d} p={v['p']:.3f}")
print("\nkey contrasts")
for k, v in S["key_contrasts"].items():
    print(f"  {k:28s} {v['mean']:+.4f} {v['wins']:2d}/{v['losses']:2d} p={v['p']:.3f}")
print("\nbayes", json.dumps(S["bayes"]))
for g, v in S["by_IR"].items():
    print(f"\nIR {g} n={v['n']}: " + ", ".join(f"{m}={v[m]:.3f}" for m in cols))
print("\nbandit pick share", S["bandit_pick_share_(hell,auc,bfisher)"], "q", S["bandit_mean_q"],
      "| winner share (main,hell,auc,bf)", S["emt_min3_winner_share_(main,hell,auc,bfisher)"])
