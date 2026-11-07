"""Phase 2, part B analysis -> summary_wrap.json and printed tables.

Pre-declared primary outcome: per dataset, the mean of the outer-test G-mean,
normalised MCC and AUC-PR of the pooled out-of-fold predictions. Baseline for
paired comparisons: acc_a99 (accuracy fitness, alpha = 0.99), the usual choice.
"""
import csv, glob, json, pickle, sys
import numpy as np
from scipy.stats import rankdata, friedmanchisquare, wilcoxon
from fsimb.stats import nogueira_stability, bayes_signed_rank

OUT = sys.argv[1] if len(sys.argv) > 1 else "results_wrap"
R = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{OUT}/*.pkl"))]
IR = {r["name"]: float(r["IR"]) for r in csv.DictReader(open("data/v2/inventory.csv"))}
M = list(R[0]["res"])
CONF = [m for m in M if "_a" in m]
FILT = [m for m in M if m not in CONF]
nd = len(R)


def mat(key):
    return np.array([[r["res"][m][key] for m in M] for r in R])


G, MC, AP, AU = mat("gmean"), mat("mcc"), mat("aucpr"), mat("auc")
P = (G + MC + AP) / 3.0                       # primary outcome
SZ = mat("size")
STAB = np.array([[nogueira_stability(r["subsets"][m], r["p"]) for m in M] for r in R])
ix = {m: i for i, m in enumerate(M)}


def holm(ps):
    o = np.argsort(ps); k = len(ps); out = np.zeros(k); run = 0.0
    for j, i in enumerate(o):
        run = max(run, min(1.0, (k - j) * ps[i])); out[i] = run
    return out


S = {"n_datasets": nd, "methods": M}
for name, A_ in (("primary", P), ("gmean", G), ("mcc", MC), ("aucpr", AP), ("auc", AU), ("size", SZ), ("stability", STAB)):
    S[f"mean_{name}"] = dict(zip(M, A_.mean(0).round(4).tolist()))
cm = [ix[m] for m in CONF]
ranks = np.array([rankdata(-row) for row in P[:, cm]])
st, p = friedmanchisquare(*P[:, cm].T)
S["friedman_configs"] = dict(stat=float(st), p=float(p), ranks=dict(zip(CONF, ranks.mean(0).round(2).tolist())))

# paired vs baselines
base = ix["acc_a99"]
for bname in ("acc_a99", "gmean_a99"):
    b = ix[bname]; rows = []
    for m in CONF + FILT:
        if m == bname:
            continue
        d = P[:, ix[m]] - P[:, b]
        pv = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        rows.append([m, float(np.median(d)), float(d.mean()), int((d > 0).sum()), int((d < 0).sum()), pv])
    adj = holm(np.array([r[5] for r in rows]))
    S[f"paired_vs_{bname}"] = [r + [float(a)] for r, a in zip(rows, adj)]
    S[f"paired_vs_{bname}_by_metric"] = {
        k: {m: float(np.median(A_[:, ix[m]] - A_[:, b])) for m in CONF if m != bname}
        for k, A_ in (("gmean", G), ("mcc", MC), ("aucpr", AP))}
S["bayes"] = {f"{a} vs {b}": dict(zip(["P(A<B)", "P(rope)", "P(A>B)"], bayes_signed_rank(P[:, ix[a]], P[:, ix[b]])))
              for a, b in (("bbrier_a99", "acc_a99"), ("bbrier_a99", "gmean_a99"), ("gmean_a99", "acc_a99"),
                           ("bbrier_a95", "acc_a99"), ("bbrier_a99", "H-top20"))}

# alpha effect
S["alpha_effect"] = {meas: {a: float(P[:, ix[f"{meas}_a{a}"]].mean()) for a in ("99", "95", "90")}
                     for meas in ("acc", "gmean", "bbrier")}
S["alpha_effect_size"] = {meas: {a: float(SZ[:, ix[f"{meas}_a{a}"]].mean()) for a in ("99", "95", "90")}
                          for meas in ("acc", "gmean", "bbrier")}

# groups
def grp_ir(n):
    return "IR<3" if IR[n] < 3 else "IR3-8" if IR[n] <= 8 else "IR>8"
def grp_min(r):
    return "min<20" if r["minority"] < 20 else "min20-40" if r["minority"] <= 40 else "min>40"
S["groups"] = {}
for key, f in (("IR", lambda r: grp_ir(r["name"])), ("minority", grp_min)):
    S["groups"][key] = {}
    for g in sorted({f(r) for r in R}):
        sel = [i for i, r in enumerate(R) if f(r) == g]
        S["groups"][key][g] = dict(n=len(sel), **{m: float(P[sel, ix[m]].mean()) for m in ("acc_a99", "gmean_a99", "mcc_a99", "bbrier_a99", "bll_a99", "auc_a99", "aucpr_a99", "H-top20", "F-top20")},
                                   diff_bbrier_vs_acc=float(np.median(P[sel, ix["bbrier_a99"]] - P[sel, ix["acc_a99"]])),
                                   wins_bbrier_vs_acc=int((P[sel, ix["bbrier_a99"]] > P[sel, ix["acc_a99"]]).sum()))
S["per_dataset_primary"] = {r["name"]: {m: round(float(P[i, ix[m]]), 3) for m in ("acc_a99", "gmean_a99", "bbrier_a99", "H-top20")}
                            for i, r in enumerate(R)}
json.dump(S, open("summary_wrap.json", "w"), indent=1)

np.set_printoptions(linewidth=170, suppress=True)
print("datasets:", nd)
print(f"\n{'config':12s} {'primary':>8s} {'gmean':>7s} {'mcc':>7s} {'aucpr':>7s} {'auc':>7s} {'size':>6s} {'stab':>6s}")
for m in M:
    i = ix[m]
    print(f"{m:12s} {P[:,i].mean():8.3f} {G[:,i].mean():7.3f} {MC[:,i].mean():7.3f} {AP[:,i].mean():7.3f} {AU[:,i].mean():7.3f} {SZ[:,i].mean():6.1f} {STAB[:,i].mean():6.3f}")
print("\nfriedman configs", json.dumps(S["friedman_configs"]))
for k in ("paired_vs_acc_a99", "paired_vs_gmean_a99"):
    print("\n", k, "(config, median diff, mean diff, wins, losses, p, holm)")
    for r in S[k]:
        print(f"  {r[0]:11s} {r[1]:+.4f} {r[2]:+.4f} {r[3]:2d}/{r[4]:2d} p={r[5]:.4f} holm={r[6]:.4f}")
print("\nbayes", json.dumps(S["bayes"]))
print("\nalpha", json.dumps(S["alpha_effect"]), "\nsize", json.dumps(S["alpha_effect_size"]))
for key, d in S["groups"].items():
    for g, v in d.items():
        print(f"\n{key} {g} n={v['n']}: " + ", ".join(f"{m}={v[m]:.3f}" for m in ("acc_a99", "gmean_a99", "bbrier_a99", "H-top20")) + f" | bbrier-acc median {v['diff_bbrier_vs_acc']:+.3f} wins {v['wins_bbrier_vs_acc']}")
