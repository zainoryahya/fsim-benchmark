"""Analysis of the review experiments (run_extra.py and run_wrap2.py). Run from the repository root.

Contrasts are paired over datasets and over families (unweighted mean of a family's tasks), with 90% bootstrap
intervals, two-sided Wilcoxon tests and equivalence within +-0.02, as in the other analyses.
Writes hdi/summary_extra.json.
"""
import glob, json, os, pickle, re
import numpy as np
from scipy.stats import wilcoxon
from fsimb.stats import nogueira_stability

ROOT = {"disc": "hdi", "held": "hdi_ho"}
CL = {"knn": ("knn",), "lrsvm": ("lr", "svm"), "rf": ("rf",)}
KS = (10, 20, 30, 40)
P3 = lambda r: float(np.mean([r["gmean"], r["mcc"], r["aucpr"]]))   # MCC is stored normalised
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


def contrast(rows):
    out = {}
    for sets, tag in ((("disc",), "disc"), (("held",), "held"), (("disc", "held"), "both")):
        sel = [(s, n, x) for s, n, x in rows if s in sets]
        if not sel:
            continue
        out[f"{tag}|dataset"] = summarize([x for *_, x in sel])
        fam = {}
        for s, n, x in sel:
            fam.setdefault((s, family(n)), []).append(x)
        out[f"{tag}|family"] = summarize([np.mean(x) for x in fam.values()])
    return out


load = lambda d: {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{d}/*.pkl"))}
S = {}
# ------------------------------------------------------------------ further baselines (run_extra)
EX = {s: load(f"{r}/results_extra") for s, r in ROOT.items()}
ev = lambda s, n, m, k, cl: float(np.mean([P3(EX[s][n]["res"][(m, k, c)]) for c in CL[cl]]))
S["extra_n"] = {s: len(v) for s, v in EX.items()}
S["extra_mean"] = {s: {f"{m}|{k}|{cl}": float(np.mean([ev(s, n, m, k, cl) for n in EX[s]]))
                       for m, ks in (("ALL", (0,)), ("RFE", KS), ("RFEw", KS), ("Hs", KS), ("Hf", KS), ("Hr", KS))
                       for k in ks for cl in CL} for s in EX if EX[s]}
S["extra_vsH"] = {}
for m in ("RFE", "RFEw", "Hf", "Hr"):
    for k in KS:
        for cl in CL:
            S["extra_vsH"][f"{m}|{k}|{cl}"] = contrast([(s, n, ev(s, n, m, k, cl) - ev(s, n, "Hs", k, cl)) for s in EX for n in EX[s]])
for k in KS:
    for cl in CL:
        S["extra_vsH"][f"ALL|{k}|{cl}"] = contrast([(s, n, ev(s, n, "ALL", 0, cl) - ev(s, n, "Hs", k, cl)) for s in EX for n in EX[s]])
# tie-break size: mean absolute and largest per-dataset change at k = 20
S["ties"] = {}
for m in ("Hf", "Hr"):
    for cl in CL:
        d = np.array([ev(s, n, m, 20, cl) - ev(s, n, "Hs", 20, cl) for s in EX for n in EX[s]])
        S["ties"][f"{m}|{cl}"] = dict(mean_abs=float(np.mean(np.abs(d))), max_abs=float(np.max(np.abs(d))) if len(d) else 0.0,
                                      share_changed=float(np.mean(np.abs(d) > 1e-12)))
S["ties"]["tied_at_k40_median"] = {s: float(np.median([np.median([i["n_tied_at_40"] for i in EX[s][n]["info"]]) for n in EX[s]])) for s in EX if EX[s]}
# tie-break sensitivity of the earlier comparisons: further baselines (results_base) and the cap-20 wrapper
# (results_cap / results_xclf) against the Hellinger filter with ANOVA-F tie-break (Hf) instead of stable order (Hs)
BASE = {s: load(f"{r}/results_base") for s, r in ROOT.items()}
CAP = {s: load(f"{r}/results_cap") for s, r in ROOT.items()}
XCLF = {s: load(f"{r}/results_xclf") for s, r in ROOT.items()}
bv = lambda s, n, m, cl: float(np.mean([P3(BASE[s][n]["res"][(m, 20, c)]) for c in CL[cl]]))
cv_ = lambda s, n, m, cl: float(np.mean([P3(CAP[s][n]["res"][m]) if c == "knn" else P3(XCLF[s][n]["res"][(m, c)]) for c in CL[cl]]))
S["tie_sens"] = {}
for ref in ("Hs", "Hf"):
    for m in ("F", "L1", "EN", "mRMR", "RFimp", "RFperm"):
        for cl in CL:
            S["tie_sens"][f"{m}-{ref}|{cl}"] = contrast([(s, n, bv(s, n, m, cl) - ev(s, n, ref, 20, cl)) for s in EX for n in EX[s]])
    for cl in CL:
        S["tie_sens"][f"cap20-{ref}|{cl}"] = contrast([(s, n, cv_(s, n, "cap20", cl) - ev(s, n, ref, 20, cl)) for s in EX for n in EX[s]])
for cl in CL:
    S["tie_sens"][f"cap20-H20own|{cl}"] = contrast([(s, n, cv_(s, n, "cap20", cl) - cv_(s, n, "H20", cl)) for s in EX for n in EX[s]])
S["tie_worst"] = {s: sorted(((float(np.mean([ev(s, n, "Hf", 20, cl) - ev(s, n, "Hs", 20, cl) for cl in ("lrsvm",)])), n) for n in EX[s]), reverse=True)[:3] for s in EX}
S["extra_stab"] ={s: {f"{m}|20": float(np.mean([nogueira_stability(EX[s][n]["subsets"][(m, 20)], EX[s][n]["p"]) for n in EX[s]]))
                       for m in ("RFE", "RFEw", "Hs", "Hf", "Hr")} for s in EX if EX[s]}
S["extra_time"] = {s: {m: float(np.median([np.median(EX[s][n]["times"][m]) for n in EX[s]])) for m in ("RFE", "RFEw")} for s in EX if EX[s]}
# ------------------------------------------------------------------ wrapper checks (run_wrap2)
W2 = {s: load(f"{r}/results_wrap2") for s, r in ROOT.items()}
wv = lambda s, n, m, cl, reps=None: float(np.mean([
    P3(W2[s][n]["res"][(m, c)]) if reps is None else np.mean([P3(W2[s][n]["res_rep"][(m, c)][r]) for r in reps])
    for c in CL[cl]]))
S["wrap2_n"] = {s: len(v) for s, v in W2.items()}
PAIRS = [("bgwo_cap20", "H20"), ("ga_cap20", "H20"), ("ga_cap20", "bgwo_cap20"), ("ga_nocap", "bgwo_nocap"),
         ("ga_nocap", "H20"), ("bgwoF_cap20", "F20"), ("bgwoF_cap20", "H20"), ("F20", "H20"), ("bgwo_nocap", "H20")]
S["wrap2"] = {}
for a, b in PAIRS:
    for cl in CL:
        S["wrap2"][f"{a}-{b}|{cl}"] = contrast([(s, n, wv(s, n, a, cl) - wv(s, n, b, cl)) for s in W2 for n in W2[s]])
for a, b in (("bgwo_cap20_x4", "bgwo_cap20"), ("bgwo_nocap_x4", "bgwo_nocap"), ("bgwo_cap20_x4", "H20")):
    for cl in CL:
        S["wrap2"][f"{a}-{b}|{cl}"] = contrast([(s, n, wv(s, n, a, cl, (0, 1)) - wv(s, n, b, cl, (0, 1))) for s in W2 for n in W2[s]])
S["wrap2_mean"] = {s: {f"{m}|{cl}": float(np.mean([wv(s, n, m, cl) for n in W2[s]]))
                       for m in ("bgwo_cap20", "bgwo_nocap", "ga_cap20", "ga_nocap", "bgwoF_cap20", "F20", "H20") for cl in CL}
                   for s in W2 if W2[s]}
S["wrap2_size"] = {s: {m: float(np.median([np.median([len(x) for x in W2[s][n]["subsets"][m]]) for n in W2[s]]))
                       for m in ("bgwo_nocap", "ga_nocap", "bgwo_nocap_x4")} for s in W2 if W2[s]}
S["wrap2_stab"] = {s: {m: float(np.mean([nogueira_stability(W2[s][n]["subsets"][m], W2[s][n]["p"]) for n in W2[s]]))
                       for m in ("bgwo_cap20", "ga_cap20", "bgwoF_cap20", "H20", "F20")} for s in W2 if W2[s]}
# search-seed stability on one fold (6 runs of bgwo_cap20 with different seeds, same data)
S["seed_stab"] = {s: float(np.mean([nogueira_stability(W2[s][n]["seed_subsets"], W2[s][n]["p"]) for n in W2[s]])) for s in W2 if W2[s]}
S["wrap2_time"] = {s: {m: float(np.median([np.median(W2[s][n]["times"][m]) for n in W2[s]]))
                       for m in ("bgwo_cap20", "ga_cap20", "bgwo_cap20_x4")} for s in W2 if W2[s]}
json.dump(S, open("hdi/summary_extra.json", "w"), indent=1)


def line(d, tag="both|family"):
    r = d.get(tag)
    return "n/a" if r is None else f"{r['mean']:+.3f}[{r['ci90'][0]:+.3f},{r['ci90'][1]:+.3f}] p={r['p']:.3f} {r['wins']}/{r['losses']}{' E' if r['equivalent'] else ''}"


print("n", S["extra_n"], S["wrap2_n"])
for key, d in S["extra_vsH"].items():
    if "|20|" in key or key.startswith("ALL|20"):
        print(f"{key:14s} disc {line(d, 'disc|dataset')}  held {line(d, 'held|dataset')}  fam {line(d)}")
print("ties", S["ties"]); print("stab", S["extra_stab"]); print("time", S["extra_time"])
for key, d in S["wrap2"].items():
    print(f"{key:28s} disc {line(d, 'disc|dataset')}  held {line(d, 'held|dataset')}  fam {line(d)}")
print("sizes", S["wrap2_size"], "stab", S["wrap2_stab"], "seed", S["seed_stab"], "time", S["wrap2_time"])
