import pickle, sys, json, numpy as np
from scipy.stats import wilcoxon
D = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "results_thr.pkl", "rb"))
names = sorted(D)
V = ["base", "prior", "tuned", "smote_fs", "fs_smote", "weighted"]
def holm(ps):
    o = np.argsort(ps); k = len(ps); res = np.zeros(k); run = 0.0
    for j, i in enumerate(o):
        run = max(run, min(1.0, (k - j) * ps[i])); res[i] = run
    return res
S = {"n": len(names)}
print("datasets:", len(names))
for fs in ("F", "H"):
    for c in ("knn", "lr"):
        vs = [v for v in V if (fs, v, c) in D[names[0]]]
        print(f"\n===== filter {fs}, classifier {c} =====\n{'variant':10s}  mcc   gmean  auc  recall")
        for v in vs:
            print(f"{v:10s} " + " ".join(f"{np.mean([D[n][(fs, v, c)][m] for n in names]):.3f}" for m in ("mcc", "gmean", "auc", "recall")))
        contr = [("prior", "base"), ("tuned", "base"), ("smote_fs", "base"), ("fs_smote", "base"),
                 ("smote_fs", "prior"), ("smote_fs", "tuned"), ("fs_smote", "prior"), ("fs_smote", "tuned"), ("tuned", "prior")]
        if c == "lr": contr += [("weighted", "prior"), ("smote_fs", "weighted")]
        for m in ("gmean", "mcc"):
            rows = []
            for a, b in contr:
                d = np.array([D[n][(fs, a, c)][m] - D[n][(fs, b, c)][m] for n in names])
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                rows.append((a, b, d.mean(), int((d > 0).sum()), int((d < 0).sum()), p))
            adj = holm(np.array([r[5] for r in rows]))
            S[f"{fs}|{c}|{m}"] = [list(r) + [float(x)] for r, x in zip(rows, adj)]
            print(f" [{m}] " + "; ".join(f"{r[0]}-{r[1]} {r[2]:+.3f} ({r[3]}/{r[4]}, h{x:.2f})" for r, x in zip(rows, adj)))
json.dump(S, open("summary_thr.json", "w"), indent=1)
