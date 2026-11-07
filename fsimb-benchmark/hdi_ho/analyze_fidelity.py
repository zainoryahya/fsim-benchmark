"""Phase 2, part A analysis -> summary_fid.json and printed tables.

Criteria (per dataset: mean over its 10 outer splits; then across datasets):
  primary    within-size-stratum Spearman(inner score, outer-test truth),
             averaged over the three truths (G-mean, MCC, AUC-PR)
  secondary  global Spearman, and the percentile of the inner-best pick
  tie rate   share of candidate pairs the inner score cannot tell apart
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import rankdata, friedmanchisquare, wilcoxon

CEILING = {"chiaretti", "CLL_SUB_111", "lymphoma", "Carcinom"}
OUT = sys.argv[1] if len(sys.argv) > 1 else "results_fid"


def corr_cols(A, B):
    """Spearman between every column of A and every column of B (rows = candidates)."""
    RA = np.apply_along_axis(rankdata, 0, A)
    RB = np.apply_along_axis(rankdata, 0, B)
    RA = RA - RA.mean(0)
    RB = RB - RB.mean(0)
    na, nb = np.sqrt((RA ** 2).sum(0)), np.sqrt((RB ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (RA.T @ RB) / (na[:, None] * nb[None, :])


def pick_percentile(S, T):
    """S: cand x F inner scores, T: cand x t truths -> F x t percentile of the inner-best pick."""
    out = np.zeros((S.shape[1], T.shape[1]))
    for f in range(S.shape[1]):
        picks = S[:, f] >= S[:, f].max() - 1e-12
        for t in range(T.shape[1]):
            v = T[picks, t].mean()
            out[f, t] = (T[:, t] < v).mean() + 0.5 * (T[:, t] == v).mean()
    return out


def tie_rate(S):
    n = S.shape[0]
    out = np.zeros(S.shape[1])
    for f in range(S.shape[1]):
        _, c = np.unique(np.round(S[:, f], 9), return_counts=True)
        out[f] = (c * (c - 1) / 2).sum() / (n * (n - 1) / 2)
    return out


def analyse_dataset(path):
    d = np.load(path, allow_pickle=False)
    I, T, sizes = d["inner"].astype(float), d["truth"].astype(float), d["sizes"]
    ns, nc, nm, ne = I.shape
    nt = T.shape[2]
    within = np.full((ns, nm * ne, nt), np.nan)
    glob_ = np.full((ns, nm * ne, nt), np.nan)
    pct_w = np.full((ns, nm * ne, nt), np.nan)
    pct_g = np.full((ns, nm * ne, nt), np.nan)
    ties = np.zeros((ns, nm * ne))
    for s in range(ns):
        S = I[s].reshape(nc, nm * ne)
        glob_[s] = corr_cols(S, T[s])
        pct_g[s] = pick_percentile(S, T[s])
        ties[s] = tie_rate(S)
        w, p = [], []
        for k in np.unique(sizes):
            ix = sizes == k
            w.append(corr_cols(S[ix], T[s][ix]))
            p.append(pick_percentile(S[ix], T[s][ix]))
        within[s] = np.nanmean(w, axis=0)
        pct_w[s] = np.mean(p, axis=0)
    r = lambda a: np.nanmean(a, axis=0).reshape(nm, ne, nt)
    return dict(name=os.path.basename(path)[:-4], n=int(d["n"]), p=int(d["p"]), minority=int(d["minority"]),
                within=r(within), glob=r(glob_), pct_within=r(pct_w), pct_glob=r(pct_g),
                ties=np.nanmean(ties, axis=0).reshape(nm, ne),
                measures=[str(x) for x in d["measures"]], estimators=[str(x) for x in d["estimators"]],
                truths=[str(x) for x in d["truths"]], truth_sd=np.nanmean(T.std(1), axis=0))


def friedman_ranks(M):
    """M: datasets x methods, higher better."""
    ranks = np.array([rankdata(-row) for row in M])
    st, p = friedmanchisquare(*M.T)
    return float(st), float(p), ranks.mean(0)


if __name__ == "__main__":
    files = sorted(glob.glob(f"{OUT}/*.npz"))
    R = [analyse_dataset(f) for f in files]
    inv = {}
    import csv
    for row in csv.DictReader(open("data/v2/inventory.csv")):
        inv[row["name"]] = float(row["IR"])
    for r in R:
        r["IR"] = inv[r["name"]]
    main = [r for r in R if r["name"] not in CEILING]
    MEAS, EST, TR = R[0]["measures"], R[0]["estimators"], R[0]["truths"]
    A = np.array([r["within"] for r in main])            # datasets x m x e x t
    prim = A.mean(axis=3)                                # datasets x m x e
    S = {"n_datasets_main": len(main), "n_datasets_all": len(R), "measures": MEAS, "estimators": EST,
         "truths": TR, "excluded_ceiling": sorted(CEILING)}

    def tab(a):
        return a.mean(0).tolist()
    S["primary_mean"] = tab(prim)
    S["global_mean"] = tab(np.array([r["glob"] for r in main]).mean(axis=3))
    S["pct_within_mean"] = tab(np.array([r["pct_within"] for r in main]).mean(axis=3))
    S["pct_glob_mean"] = tab(np.array([r["pct_glob"] for r in main]).mean(axis=3))
    S["tie_rate_mean"] = tab(np.array([r["ties"] for r in main]))
    S["primary_by_truth"] = {t: A[..., i].mean(0).tolist() for i, t in enumerate(TR)}

    # ranks of measures within each estimator, and of estimators within each measure
    S["friedman_measures"] = {}
    for ei, e in enumerate(EST):
        st, p, rk = friedman_ranks(prim[:, :, ei])
        S["friedman_measures"][e] = dict(stat=st, p=p, ranks=dict(zip(MEAS, rk.tolist())))
    st, p, rk = friedman_ranks(prim.mean(axis=2))
    S["friedman_measures_avg_over_estimators"] = dict(stat=st, p=p, ranks=dict(zip(MEAS, rk.tolist())))
    S["friedman_estimators_avg_over_measures"] = {}
    st, p, rk = friedman_ranks(prim.mean(axis=1))
    S["friedman_estimators_avg_over_measures"] = dict(stat=st, p=p, ranks=dict(zip(EST, rk.tolist())))

    # all 48 combos ranked together
    flat = prim.reshape(len(main), -1)
    names = [f"{m}|{e}" for m in MEAS for e in EST]
    st, p, rk = friedman_ranks(flat)
    o = np.argsort(rk)
    S["friedman_all48"] = dict(stat=st, p=p, top10=[[names[i], float(rk[i]), float(flat[:, i].mean())] for i in o[:10]],
                               bottom5=[[names[i], float(rk[i]), float(flat[:, i].mean())] for i in o[-5:]])

    # family x truth (does each family win on its own metric?)
    fam = {"label": [MEAS.index(m) for m in MEAS[:8]], "prob": [MEAS.index(m) for m in MEAS[8:]]}
    S["family_by_truth"] = {f: {t: float(A[:, ix, :, i].mean()) for i, t in enumerate(TR)} for f, ix in fam.items()}

    # groups
    def grp(r):
        return ("IR<3" if r["IR"] < 3 else "IR3-8" if r["IR"] <= 8 else "IR>8")
    def grm(r):
        return ("min<20" if r["minority"] < 20 else "min20-40" if r["minority"] <= 40 else "min>40")
    for key, f in (("by_IR", grp), ("by_minority", grm)):
        S[key] = {}
        for g in sorted({f(r) for r in main}):
            sub = np.array([x for x, r in zip(prim, main) if f(r) == g])
            S[key][g] = dict(n=len(sub), measure_mean_over_est=dict(zip(MEAS, sub.mean(axis=2).mean(0).tolist())),
                             estimator_mean_over_meas=dict(zip(EST, sub.mean(axis=1).mean(0).tolist())),
                             best=names[int(np.argmax(sub.reshape(len(sub), -1).mean(0)))])
    # paired Wilcoxon: best combo vs the common choices
    best = int(np.argmax(flat.mean(0)))
    S["best_combo"] = names[best]
    ref = {"acc|loo": names.index("acc|loo"), "gmean|loo": names.index("gmean|loo")}
    S["best_vs_common"] = {}
    for k, j in ref.items():
        dlt = flat[:, best] - flat[:, j]
        S["best_vs_common"][k] = dict(median_diff=float(np.median(dlt)), wins=int((dlt > 0).sum()), n=len(dlt),
                                      p=float(wilcoxon(dlt).pvalue) if np.any(dlt != 0) else 1.0)
    S["per_dataset"] = {r["name"]: dict(IR=r["IR"], minority=r["minority"], n=r["n"], p=r["p"],
                                        best=names[int(np.argmax(r["within"].mean(axis=2).ravel()))],
                                        acc_loo=float(r["within"].mean(axis=2)[0, 0]),
                                        gmean_loo=float(r["within"].mean(axis=2)[2, 0]))
                        for r in R}
    json.dump(S, open("summary_fid.json", "w"), indent=1)

    np.set_printoptions(linewidth=170, suppress=True)
    print("main datasets:", len(main), "| excluded:", sorted(CEILING))
    print("\nPRIMARY (within-size Spearman, mean of 3 truths) rows=measures cols=", EST)
    for m, row in zip(MEAS, prim.mean(0)):
        print(f"{m:7s}", " ".join(f"{v:7.3f}" for v in row), f"| mean {row.mean():.3f}")
    print("\nglobal Spearman"); [print(f"{m:7s}", " ".join(f"{v:7.3f}" for v in row)) for m, row in zip(MEAS, np.array(S['global_mean']))]
    print("\npick percentile within size"); [print(f"{m:7s}", " ".join(f"{v:7.3f}" for v in row)) for m, row in zip(MEAS, np.array(S['pct_within_mean']))]
    print("\ntie rate"); [print(f"{m:7s}", " ".join(f"{v:7.3f}" for v in row)) for m, row in zip(MEAS, np.array(S['tie_rate_mean']))]
    for k in ["friedman_measures_avg_over_estimators", "friedman_estimators_avg_over_measures", "friedman_all48", "family_by_truth", "best_vs_common"]:
        print("\n", k, json.dumps(S[k]))
    for k in ["by_IR", "by_minority"]:
        for g, v in S[k].items():
            print(f"\n{k} {g} n={v['n']} best={v['best']}\n  measures: {v['measure_mean_over_est']}\n  estimators: {v['estimator_mean_over_meas']}")
