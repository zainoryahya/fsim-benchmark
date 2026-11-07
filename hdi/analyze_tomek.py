"""Tomek / SMOTE+Tomek versus thresholds: means and paired contrasts (discovery, held-out, pooled; dataset and
family units; 90% bootstrap CI; equivalence within +-0.02). Run from the repository root. Writes hdi/summary_tomek.json."""
import json, pickle, re, numpy as np
from scipy.stats import wilcoxon

R = {"disc": pickle.load(open("hdi/results_tomek.pkl", "rb")), "held": pickle.load(open("hdi_ho/results_tomek.pkl", "rb"))}
THR = {"disc": pickle.load(open("hdi/results_thr.pkl", "rb")), "held": pickle.load(open("hdi_ho/results_thr.pkl", "rb"))}
V = ["base", "prior", "tomek_fs", "fs_tomek", "smtk_fs", "fs_smtk"]
CONF = [(fs, c) for fs in "FH" for c in ("knn", "lr")]
rng = np.random.default_rng(0)


def family(n):
    m = re.match(r"^(gisette|BASEHOCK|SMK_CAN_187|arcene|PCMAC|RELATHE|singh|chowdary)(_ir\d+)?$", n)
    return m.group(1) if m else n


def summarize(v, margin=0.02):
    v = np.asarray(v)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [5, 95])
    p = float(wilcoxon(v).pvalue) if np.any(v != 0) else 1.0
    return dict(n=len(v), mean=float(v.mean()), ci90=[float(lo), float(hi)], p=p, wins=int((v > 0).sum()),
                losses=int((v < 0).sum()), equivalent=bool(lo > -margin and hi < margin))


val = lambda s, n, v, m: float(np.mean([R[s][n][(fs, v, c)][m] for fs, c in CONF]))
S = {"means": {}, "removed": {}, "contrasts": {}}
for s in R:
    names = sorted(R[s])
    S["means"][s] = {v: {m: float(np.mean([val(s, n, v, m) for n in names])) for m in ("gmean", "mcc", "auc")} for v in V}
    S["removed"][s] = {k: float(np.mean([R[s][n]["removed"][k] for n in names])) for k in ("full", "selected")}
    # consistency: prior recomputed here equals prior in results_thr (same folds and seed)
    S.setdefault("prior_check_maxdiff", {})[s] = float(max(abs(val(s, n, "prior", "gmean") - np.mean(
        [THR[s][n][(fs, "prior", c)]["gmean"] for fs, c in CONF])) for n in names))
CON = [("tomek_fs", "base"), ("fs_tomek", "base"), ("tomek_fs", "prior"), ("fs_tomek", "prior"),
       ("smtk_fs", "prior"), ("fs_smtk", "prior")]
for a, b in CON:
    for m in ("gmean", "mcc"):
        key = f"{a} - {b} | {m}"; S["contrasts"][key] = {}
        rows = [(s, n, val(s, n, a, m) - val(s, n, b, m)) for s in R for n in sorted(R[s])]
        for sets, tag in ((("disc",), "disc"), (("held",), "held"), (("disc", "held"), "both")):
            sel = [(s, n, x) for s, n, x in rows if s in sets]
            S["contrasts"][key][f"{tag}|dataset"] = summarize([x for _, _, x in sel])
            fam = {}
            for s, n, x in sel: fam.setdefault((s, family(n)), []).append(x)
            S["contrasts"][key][f"{tag}|family"] = summarize([np.mean(x) for x in fam.values()])
json.dump(S, open("hdi/summary_tomek.json", "w"), indent=1)
for s in R:
    print(s, "removed", {k: round(v, 3) for k, v in S["removed"][s].items()}, "prior check", S["prior_check_maxdiff"][s])
    for v in V:
        print(f"  {v:9s} " + " ".join(f"{m}={S['means'][s][v][m]:.3f}" for m in ("gmean", "mcc", "auc")))
for k, d in S["contrasts"].items():
    print(f"{k:28s} " + "  ".join(f"{t}:{d[t]['mean']:+.3f}[{d[t]['ci90'][0]:+.3f},{d[t]['ci90'][1]:+.3f}]{'E' if d[t]['equivalent'] else ''}"
                                   for t in ("disc|dataset", "held|dataset", "both|family")))
