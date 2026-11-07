import pickle, json, csv, numpy as np
from collections import defaultdict
from scipy.stats import wilcoxon
from run_leak2 import PIPES, FS, CLF
out = pickle.load(open("results_leak2.pkl", "rb"))
real = {n: r for n, k, r in out if k < 0}
null = defaultdict(list)
for n, k, r in out:
    if k >= 0: null[n].append(r)
names = sorted(real)
def holm(ps):
    o = np.argsort(ps); k = len(ps); res = np.zeros(k); run = 0.0
    for j, i in enumerate(o):
        run = max(run, min(1.0, (k - j) * ps[i])); res[i] = run
    return res
S = {"n_real": len(names), "null_datasets": sorted(null), "n_perm": len(next(iter(null.values())))}
print(f"real datasets: {len(names)}, null datasets: {len(null)} x {S['n_perm']} permutations")
contr = [("SMOTE->FS", "FS only"), ("FS->SMOTE", "FS only"), ("SMOTE->FS", "FS->SMOTE"), ("leak SMOTE-first", "FS only"), ("leak FS-first", "FS only")]
for fs in FS:
    for c in CLF:
        print(f"\n===== filter {fs}, classifier {c} =====")
        print(f"{'pipeline':18s} | real: mcc   auc  gmean | null: mcc    auc  gmean  P(auc>0.6)")
        for pipe in PIPES:
            rm = {m: np.mean([real[n][(fs, pipe, c)][m] for n in names]) for m in ("mcc", "auc", "gmean")}
            nm = {m: np.mean([np.mean([r[(fs, pipe, c)][m] for r in null[n]]) for n in null]) for m in ("mcc", "auc", "gmean")}
            fa = np.mean([np.mean([r[(fs, pipe, c)]["auc"] > 0.6 for r in null[n]]) for n in null])
            S[f"{fs}|{c}|{pipe}"] = dict(real=rm, null=nm, null_auc_gt_06=float(fa))
            print(f"{pipe:18s} |      {rm['mcc']:.3f} {rm['auc']:.3f} {rm['gmean']:.3f} |      {nm['mcc']:+.3f} {nm['auc']:.3f} {nm['gmean']:.3f}  {fa:.3f}")
        for m in ("mcc", "auc", "gmean"):
            rows = []
            for a, b in contr:
                d = np.array([real[n][(fs, a, c)][m] - real[n][(fs, b, c)][m] for n in names])
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                rows.append([a, b, float(d.mean()), int((d > 0).sum()), int((d < 0).sum()), p])
            adj = holm(np.array([r[5] for r in rows]))
            S[f"{fs}|{c}|contrasts|{m}"] = [r + [float(x)] for r, x in zip(rows, adj)]
            print(f"  [{m}] " + "; ".join(f"{r[0]} - {r[1]}: {r[2]:+.3f} ({r[3]}/{r[4]}, holm {x:.3f})" for r, x in zip(rows, adj)))
json.dump(S, open("summary_leak2.json", "w"), indent=1)
