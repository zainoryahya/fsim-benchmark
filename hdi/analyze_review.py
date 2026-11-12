"""Additional analyses requested in review, from the stored results (no new runs). Run from the repository root.

1. Imbalance moderation: does the advantage of the Hellinger filter (k = 20) over each further method, or the
   cap-20 wrapper's advantage over it, change with the imbalance ratio?  Within each controlled-imbalance family
   (2-4 IR variants) the slope of the difference on log2(IR) is computed; slopes are summarised over families, and
   the Spearman correlation of the difference with IR is reported over all 52 tasks.
2. Sparse binary text tasks (BASEHOCK, PCMAC, RELATHE; 8 tasks): further methods minus Hellinger, k = 10..40.
3. Components of the primary outcome (G-mean, normalised MCC as stored, AUC-PR) for the k = 20 baseline contrasts and the
   cap-20 wrapper, pooled over 38 families.
4. AUC-ROC in the threshold experiment: SMOTE (before or after the filter) minus no resampling.
5. Subset sizes of the uncapped wrapper and the EMT variants.
Writes hdi/summary_review.json.
"""
import csv, glob, json, os, pickle, re
import numpy as np
from scipy.stats import wilcoxon, spearmanr

ROOT = {"disc": "hdi", "held": "hdi_ho"}
CL = {"knn": ("knn",), "lrsvm": ("lr", "svm"), "rf": ("rf",)}
rng = np.random.default_rng(0)


def family(n):
    m = re.match(r"^(gisette|BASEHOCK|SMK_CAN_187|arcene|PCMAC|RELATHE|singh|chowdary)(_ir\d+)?$", n)
    return m.group(1) if m else n


def summarize(v, margin=0.02):
    v = np.asarray(v, float)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)])
    lo, hi = np.percentile(bs, [5, 95])
    p = float(wilcoxon(v).pvalue) if np.any(v != 0) else 1.0
    return dict(n=len(v), mean=float(v.mean()), ci90=[float(lo), float(hi)], p=p, wins=int((v > 0).sum()),
                losses=int((v < 0).sum()), equivalent=bool(lo > -margin and hi < margin))


def fam_pool(rows):
    """rows: list of (suite, task, value) -> per-family means (suite, family)."""
    f = {}
    for s, n, x in rows:
        f.setdefault((s, family(n)), []).append(x)
    return [float(np.mean(v)) for v in f.values()]


IR = {}
for s, r in ROOT.items():
    for row in csv.DictReader(open(f"{r}/data/v2/inventory.csv")):
        IR[(s, row["name"])] = float(row["IR"])
load = lambda d: {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{d}/*.pkl"))}
BASE = {s: load(f"{r}/results_base") for s, r in ROOT.items()}
CAP = {s: load(f"{r}/results_cap") for s, r in ROOT.items()}
XCLF = {s: load(f"{r}/results_xclf") for s, r in ROOT.items()}
S = {}


def bval(s, n, m, k, cl, q=None):
    rs = [BASE[s][n]["res"][(m, k, c)] for c in CL[cl]]
    if q is None:
        return float(np.mean([np.mean([r["gmean"], r["mcc"], r["aucpr"]]) for r in rs]))
    return float(np.mean([r[q] for r in rs]))


def wval(s, n, m, cl, q=None):
    """cap-20 wrapper or H20 of the cap run; 5-NN from results_cap, others from results_xclf."""
    out = []
    for c in CL[cl]:
        r = CAP[s][n]["res"][m] if c == "knn" else XCLF[s][n]["res"][(m, c)]
        out.append(np.mean([r["gmean"], r["mcc"], r["aucpr"]]) if q is None else r[q])
    return float(np.mean(out))


METH = ["F", "L1", "EN", "mRMR", "RFimp", "RFperm"]
# 1. imbalance moderation -------------------------------------------------------------------------------------
mod = {}
contrasts = {f"H-{m}": (lambda s, n, cl, m=m: bval(s, n, "H", 20, cl) - bval(s, n, m, 20, cl)) for m in METH}
contrasts["cap20-H20"] = lambda s, n, cl: wval(s, n, "cap20", cl) - wval(s, n, "H20", cl)
for name, fn in contrasts.items():
    for cl in CL:
        rows = [(s, n, fn(s, n, cl)) for s in BASE for n in BASE[s]]
        x = np.log2([IR[(s, n)] for s, n, _ in rows]); v = np.array([d for *_, d in rows])
        rho, prho = spearmanr(x, v)
        fams = {}
        for (s, n, d), xi in zip(rows, x):
            fams.setdefault((s, family(n)), []).append((xi, d))
        slopes = {f"{s}:{f}": float(np.polyfit(*zip(*pts), 1)[0]) for (s, f), pts in fams.items() if len(pts) >= 2}
        sl = np.array(list(slopes.values()))
        mod[f"{name}|{cl}"] = dict(spearman_all=float(rho), p_spearman=float(prho), family_slopes=slopes,
                                   mean_slope=float(sl.mean()), n_pos=int((sl > 0).sum()), n_neg=int((sl < 0).sum()))
S["ir_moderation"] = mod
# 2. sparse binary text tasks ----------------------------------------------------------------------------------
TEXT = [(s, n) for s in BASE for n in BASE[s] if family(n) in ("BASEHOCK", "PCMAC", "RELATHE")]
sp = {}
for m in METH:
    for k in (10, 20, 30, 40):
        for cl in CL:
            v = [bval(s, n, m, k, cl) - bval(s, n, "H", k, cl) for s, n in TEXT]
            sp[f"{m}|{k}|{cl}"] = dict(mean=float(np.mean(v)), wins=int(sum(x > 0 for x in v)), losses=int(sum(x < 0 for x in v)))
S["text_tasks"] = dict(n=len(TEXT), tasks=[f"{s}:{n}" for s, n in TEXT], diff=sp)
# 3. components ----------------------------------------------------------------------------------------------
comp = {}
for q in ("gmean", "mcc", "aucpr"):
    for m in METH:
        for cl in CL:
            rows = [(s, n, bval(s, n, m, 20, cl, q) - bval(s, n, "H", 20, cl, q)) for s in BASE for n in BASE[s]]
            comp[f"{m}-H|{q}|{cl}"] = summarize(fam_pool(rows))
    for cl in CL:
        rows = [(s, n, wval(s, n, "cap20", cl, q) - wval(s, n, "H20", cl, q)) for s in BASE for n in BASE[s]]
        comp[f"cap20-H20|{q}|{cl}"] = summarize(fam_pool(rows))
S["components"] = comp
# 4. AUC in the threshold experiment -------------------------------------------------------------------------
THR = {s: pickle.load(open(f"{r}/results_thr.pkl", "rb")) for s, r in ROOT.items()}
auc = {}
for v in ("smote_fs", "fs_smote"):
    rows = [(s, n, np.mean([THR[s][n][(fs, v, c)]["auc"] - THR[s][n][(fs, "base", c)]["auc"] for fs in "FH" for c in ("knn", "lr")]))
            for s in THR for n in THR[s] if n in BASE[s]]
    auc[v] = dict(dataset=summarize([x for *_, x in rows]), family=summarize(fam_pool(rows)))
S["threshold_auc"] = auc
# 5. subset sizes -------------------------------------------------------------------------------------------
sizes = {}
for s, r in ROOT.items():
    W = load(f"{r}/results_wrapH"); E = load(f"{r}/results_emt")
    sizes[s] = {"wrapper_nocap": float(np.median([np.median([len(x) for x in W[n]["subsets"]["bbrier_a99"]]) for n in W])),
                **{m: float(np.median([np.median([len(x) for x in E[n]["subsets"][m]]) for n in E]))
                   for m in ("emt_f", "emt_relief", "emt_min3", "emt_min3_rl", "emt_rand3")}}
    sizes[s]["wrapper_nocap_range"] = [float(np.min([np.median([len(x) for x in W[n]["subsets"]["bbrier_a99"]]) for n in W])),
                                       float(np.max([np.median([len(x) for x in W[n]["subsets"]["bbrier_a99"]]) for n in W]))]
S["sizes"] = sizes
json.dump(S, open("hdi/summary_review.json", "w"), indent=1)

print("== IR moderation (Spearman over 52 tasks; mean within-family slope per doubling of IR, +/- families)")
for k, d in mod.items():
    print(f"  {k:16s} rho={d['spearman_all']:+.2f} (p={d['p_spearman']:.2f})  slope={d['mean_slope']:+.4f}  {d['n_pos']}+/{d['n_neg']}-")
print("== text tasks (n=%d): method - H" % len(TEXT))
for k in (10, 20, 30, 40):
    print(f"  k={k}: " + "  ".join(f"{m}:" + "/".join(f"{sp[f'{m}|{k}|{cl}']['mean']:+.3f}" for cl in CL) for m in METH))
print("== components (family level, method - H, k=20)")
for q in ("gmean", "mcc", "aucpr"):
    print(f"  {q}: " + "  ".join(f"{m}:" + "/".join(f"{comp[(f'{m}-H|{q}|{cl}' if m != 'cap20' else f'cap20-H20|{q}|{cl}')]['mean']:+.3f}" for cl in CL) for m in METH + ["cap20"]))
print("== threshold AUC (SMOTE - none):", {v: (round(a['dataset']['mean'], 4), round(a['family']['mean'], 4), round(a['family']['p'], 3)) for v, a in auc.items()})
print("== sizes:", sizes)
