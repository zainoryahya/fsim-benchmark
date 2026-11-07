"""Held-out validation: tests the pre-registered hypotheses H1-H6 (see project document)."""
import glob, os, pickle, json, numpy as np
from scipy.stats import wilcoxon

def load(d): return {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{d}/*.pkl"))}
W, WH, C, E, X, Rg = (load(d) for d in ("results_wrap", "results_wrapH", "results_cap", "results_emt", "results_xclf", "results_ridge"))
names = sorted(C)
print("held-out datasets:", len(names), names)

def prim(res, keys=("gmean", "mcc", "aucpr")): return np.mean([res[k] for k in keys])
def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    return dict(mean=float(d.mean()), median=float(np.median(d)), wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                p=p, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
def holm(ps):
    o = np.argsort(ps); k = len(ps); out = np.zeros(k); run = 0.0
    for j, i in enumerate(o):
        run = max(run, min(1.0, (k - j) * ps[i])); out[i] = run
    return out
def show(label, r, extra=""):
    print(f"  {label:48s} {r['mean']:+.4f} (95% CI {r['ci'][0]:+.3f},{r['ci'][1]:+.3f}) med {r['median']:+.4f} {r['wins']}/{r['losses']} p={r['p']:.3f} {extra}")

def run(sel, tag):
    print(f"\n########## {tag}: n={len(sel)} ##########")
    out = {}
    # H1
    a = [prim(WH[n]["res"]["bbrier_a99"]) for n in sel]; b = [prim(WH[n]["res"]["acc_a99"]) for n in sel]
    out["H1"] = paired(a, b); print("H1 bbrier vs acc (H pool, 5-NN, uncapped)"); show("bbrier_a99 - acc_a99", out["H1"])
    # H2
    out["H2"] = paired([prim(C[n]["res"]["cap20"]) for n in sel], [prim(C[n]["res"]["H20"]) for n in sel])
    print("H2 cap20 vs H20, final 5-NN"); show("cap20 - H20", out["H2"])
    # H3
    def clf_prim(n, m, cl): return np.mean([X[n]["res"][(m, cl)][k] for k in ("gmean", "mcc", "aucpr")])
    out["H3a"] = paired([np.mean([clf_prim(n, "cap20", c) for c in ("lr", "svm")]) for n in sel],
                        [np.mean([clf_prim(n, "H20", c) for c in ("lr", "svm")]) for n in sel])
    print("H3a cap20(5-NN search) vs H20, final LR/SVM avg (criterion: mean < 0.02)"); show("cap20 - H20", out["H3a"])
    def rprim(n, m, cl): return np.mean([Rg[n]["res"][(m, cl)][k] for k in ("gmean", "mcc", "aucpr")])
    out["H3b"] = paired([np.mean([rprim(n, "ridge_cap20", c) for c in ("rls", "lr", "svm")]) for n in sel],
                        [np.mean([rprim(n, "H20", c) for c in ("rls", "lr", "svm")]) for n in sel])
    print("H3b ridge_cap20 vs H20, final ridge/LR/SVM avg (criterion: not > 0)"); show("ridge_cap20 - H20", out["H3b"])
    # H4
    out["H4a"] = paired([prim(E[n]["res"]["emt_min3"]) for n in sel], [prim(C[n]["res"]["cap20"]) for n in sel])
    out["H4b"] = paired([prim(E[n]["res"]["emt_min3"]) for n in sel], [prim(WH[n]["res"]["bbrier_a99"]) for n in sel])
    print("H4 EMT-min3 (5-NN)"); show("emt_min3 - cap20", out["H4a"]); show("emt_min3 - uncapped single (bbrier_a99)", out["H4b"])
    keys = ["H2", "H3a", "H3b", "H4a", "H4b"]
    adj = holm(np.array([out[k]["p"] for k in keys]))
    for k, a_ in zip(keys, adj): out[k]["holm"] = float(a_)
    print("  Holm (H2,H3a,H3b,H4a,H4b):", {k: round(float(a_), 3) for k, a_ in zip(keys, adj)})
    return out

S = {"n": len(names), "all": run(names, "ALL")}
raw_mcc = {n: 2 * C[n]["res"]["H20"]["mcc"] - 1 for n in names}
sens = [n for n in names if raw_mcc[n] < 0.95]
print("\nceiling (H20 MCC >= 0.95):", [n for n in names if n not in sens])
S["sensitivity"] = run(sens, "SENSITIVITY (ceiling excluded)")

# H5
if os.path.exists("results_leak2.pkl"):
    from collections import defaultdict
    out = pickle.load(open("results_leak2.pkl", "rb"))
    null = defaultdict(list); real = {}
    for n, k, r in out:
        (real.__setitem__(n, r) if k < 0 else null[n].append(r))
    print("\n##### H5 leakage (null MCC, mean over datasets of mean over permutations) #####")
    S["H5"] = {}
    for fs in ("F", "H"):
        for c in ("knn", "lr"):
            row = {}
            for pipe in ("FS only", "SMOTE->FS", "FS->SMOTE", "leak SMOTE-first", "leak FS-first"):
                row[pipe] = float(np.mean([np.mean([r[(fs, pipe, c)]["mcc"] for r in null[n]]) for n in null]))
            S["H5"][f"{fs}|{c}"] = row
            ok1 = all(abs(row[p]) <= 0.03 for p in ("FS only", "SMOTE->FS", "FS->SMOTE")); ok2 = row["leak SMOTE-first"] > 0.20
            print(f"  {fs}/{c}: " + ", ".join(f"{k} {v:+.3f}" for k, v in row.items()) + f" | correct within +-0.03: {ok1}; SMOTE-first > 0.20: {ok2}")
# H6
if os.path.exists("results_thr.pkl"):
    T = pickle.load(open("results_thr.pkl", "rb"))
    print("\n##### H6 prior threshold vs SMOTE (mean over 4 filter x classifier combos) #####")
    S["H6"] = {}
    for v in ("smote_fs", "fs_smote"):
        g = np.mean([np.mean([T[n][(fs, v, c)]["gmean"] - T[n][(fs, "prior", c)]["gmean"] for n in names]) for fs in "FH" for c in ("knn", "lr")])
        m = np.mean([np.mean([T[n][(fs, v, c)]["mcc"] - T[n][(fs, "prior", c)]["mcc"] for n in names]) for fs in "FH" for c in ("knn", "lr")])
        S["H6"][v] = dict(gmean_diff=float(g), mcc_diff=float(m))
        print(f"  {v} - prior: G-mean {g:+.4f} (|.|<0.03: {abs(g) < 0.03}), MCC {m:+.4f} (|.|<0.02: {abs(m) < 0.02})")
json.dump(S, open("summary_ho.json", "w"), indent=1)
