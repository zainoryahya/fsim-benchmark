"""Summarise results_main.pkl: metrics, statistics, stability, recovery."""
import pickle, json, numpy as np
from fsimb.stats import nogueira_stability, friedman, nemenyi_cd, holm_wilcoxon, bayes_signed_rank

R = pickle.load(open("results_main.pkl", "rb"))
M, res = R["methods"], R["results"]
D = list(res)
metr = ["gmean", "mcc", "aucpr", "recall_min", "nfeat"]
T = {k: np.array([[np.mean(res[d][0][m][k]) for m in M] for d in D]) for k in metr}
STAB = np.array([[nogueira_stability(res[d][0][m]["subsets"], res[d][1]["p"]) for m in M] for d in D])

summary = {"datasets": D, "methods": M}
for k in metr:
    summary[k] = T[k].round(3).tolist()
summary["stability"] = STAB.round(3).tolist()

st, p, ranks = friedman(T["gmean"])
summary["friedman_gmean"] = {"stat": st, "p": p, "mean_ranks": dict(zip(M, ranks.round(2).tolist())),
                             "nemenyi_cd": float(nemenyi_cd(len(M), len(D)))}
st2, p2, ranks2 = friedman(T["mcc"])
summary["friedman_mcc"] = {"stat": st2, "p": p2, "mean_ranks": dict(zip(M, ranks2.round(2).tolist()))}
st3, p3, ranks3 = friedman(STAB)
summary["friedman_stability"] = {"stat": st3, "p": p3, "mean_ranks": dict(zip(M, ranks3.round(2).tolist()))}
summary["holm_vs_RL"] = holm_wilcoxon(T["gmean"], M, "EMT-MinAware-RL")

g = lambda m: T["gmean"][:, M.index(m)]
pairs = [("EMT-MinAware-RL", "EMT-MinAware"), ("EMT-MinAware-RL", "GWO-Gm"),
         ("EMT-MinAware", "EMT-Relief"), ("GWO-Gm", "GWO-Acc"), ("EMT-MinAware-RL", "F-top20")]
summary["bayes"] = {f"{a} vs {b}": dict(zip(["P(A<B)", "P(rope)", "P(A>B)"], bayes_signed_rank(g(a), g(b))))
                    for a, b in pairs}

# H1: accuracy vs G-mean fitness
ia, ig = M.index("GWO-Acc"), M.index("GWO-Gm")
summary["H1"] = {"gmean_wins": int((T["gmean"][:, ig] > T["gmean"][:, ia]).sum()),
                 "gmean_ties": int((T["gmean"][:, ig] == T["gmean"][:, ia]).sum()),
                 "recall_wins": int((T["recall_min"][:, ig] > T["recall_min"][:, ia]).sum()),
                 "mean_recall_acc": float(T["recall_min"][:, ia].mean()),
                 "mean_recall_gm": float(T["recall_min"][:, ig].mean())}

# synthetic recovery precision
rec = {}
for d in D:
    rel = res[d][1]["relevant"]
    if rel is None:
        continue
    rel = set(np.asarray(rel).tolist())
    rec[d] = {m: float(np.mean([len(rel & set(s.tolist())) / max(1, len(s)) for s in res[d][0][m]["subsets"]]))
              for m in M}
summary["recovery_precision"] = rec

# bandit behaviour
names = ["hellinger", "auc", "bfisher"]
picks = np.zeros(3)
for d in D:
    for inf in res[d][0]["EMT-MinAware-RL"]["info"]:
        picks += inf["picks"]
summary["bandit_pick_share"] = dict(zip(names, (picks / picks.sum()).round(3).tolist()))
summary["secs"] = {d: round(res[d][1]["secs"]) for d in D}

json.dump(summary, open("summary_main.json", "w"), indent=1, ensure_ascii=False)
np.set_printoptions(linewidth=160)
print("methods", M)
for k in ["gmean", "mcc", "recall_min", "nfeat"]:
    print("\n", k)
    for d, row in zip(D, T[k]):
        print(f"{d:14s}", " ".join(f"{v:6.3f}" for v in row))
print("\nstability")
for d, row in zip(D, STAB):
    print(f"{d:14s}", " ".join(f"{v:6.3f}" for v in row))
for k in ["friedman_gmean", "friedman_mcc", "friedman_stability", "holm_vs_RL", "bayes", "H1", "recovery_precision", "bandit_pick_share", "secs"]:
    print("\n", k, json.dumps(summary[k], ensure_ascii=False))
