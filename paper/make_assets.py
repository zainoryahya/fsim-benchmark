"""Builds every table (tables/*.tex) and figure (figs/*.pdf) of the paper from the saved results.

Discovery = /home/claude/hdi (34 non-ceiling sets), held-out = /home/claude/hdi_ho (18 sets).
Primary outcome in the wrapper/filter results = mean(G-mean, (MCC+1)/2, AUC-PR) on pooled out-of-fold
predictions; the leakage and threshold experiments report raw MCC (chance level 0).
"""
import csv, glob, json, os, pickle
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, H, OUT = "/home/claude/hdi", "/home/claude/hdi_ho", "/home/claude/paper"
os.makedirs(f"{OUT}/tables", exist_ok=True); os.makedirs(f"{OUT}/figs", exist_ok=True)
CEILING = {"chiaretti", "CLL_SUB_111", "lymphoma", "Carcinom"}
J = lambda p: json.load(open(p))
facts = {}


def load(d):
    return {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{d}/*.pkl"))}


def prim(r):
    return np.mean([r[k] for k in ("gmean", "mcc", "aucpr")])


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return d.mean(), int((d > 0).sum()), int((d < 0).sum()), p


def f3(x, sign=False):
    x = round(float(x), 3) + 0.0          # avoid printing a negative zero
    if sign and x == 0:
        return "0.000"                    # no sign on a value that rounds to zero
    return (f"{x:+.3f}" if sign else f"{x:.3f}").replace("-", "$-$")


# ---------------------------------------------------------------- Table 1: inventory
inv = []
for path, tag in ((f"{D}/data/v2/inventory.csv", "D"), (f"{H}/data/v2/inventory.csv", "H")):
    for r in csv.DictReader(open(path)):
        r["set"] = tag
        inv.append(r)
kinds = {"natural": "nat.", "controlled": "contr.", "synthetic": "synth."}
IMAGE = {"orlraws10P", "pixraw10P", "warpAR10P", "warpPIE10P"}
lines = []
for r in inv:
    name = r["name"].replace("_", r"\_")
    flag = r"$^{\dagger}$" if r["name"] in CEILING else ""
    lines.append(f"{name}{flag} & {'Discovery' if r['set']=='D' else 'Held-out'} & {'image' if r['name'] in IMAGE else kinds[r['kind']]} & "
                 f"{r['n']} & {r['p']} & {r['minority']} & {float(r['IR']):.2f} \\\\")
open(f"{OUT}/tables/inventory.tex", "w").write("\n".join(lines) + "\n")
used = [r for r in inv if r["name"] not in CEILING]
for tag in "DH":
    s = [r for r in used if r["set"] == tag]
    facts[f"inv_{tag}"] = dict(n=len(s), median_n=float(np.median([int(r["n"]) for r in s])),
                               p=[int(min(int(r["p"]) for r in s)), int(max(int(r["p"]) for r in s))],
                               IR=[min(float(r["IR"]) for r in s), max(float(r["IR"]) for r in s)],
                               min_minority=min(int(r["minority"]) for r in s),
                               kinds={k: sum(r["kind"] == k for r in s) for k in kinds})
# expected minority count per outer test fold (5-fold)
facts["median_test_minority"] = float(np.median([int(r["minority"]) / 5 for r in used]))

# ---------------------------------------------------------------- Table 2: leakage
PIPES = ["FS only", "SMOTE->FS", "FS->SMOTE", "leak SMOTE-first", "leak FS-first"]
PNAME = {"FS only": "FS in fold", "SMOTE->FS": r"SMOTE$\to$FS in fold", "FS->SMOTE": r"FS$\to$SMOTE in fold",
         "leak SMOTE-first": "SMOTE before CV (leaky)", "leak FS-first": "FS before CV (leaky)"}
CONF = ["F|knn", "F|lr", "H|knn", "H|lr"]
L = {"D": J(f"{D}/summary_leak2.json"), "H": J(f"{H}/summary_leak2.json")}
rows = []
for pipe in PIPES:
    cells = []
    for s in "DH":
        nm = [L[s][f"{c}|{pipe}"]["null"]["mcc"] for c in CONF]
        na = [L[s][f"{c}|{pipe}"]["null"]["auc"] for c in CONF]
        if pipe == "FS only":
            dm = "--"
        else:
            vals = []
            for c in CONF:
                for a, b, mean, *_ in L[s][f"{c}|contrasts|mcc"]:
                    if a == pipe and b == "FS only":
                        vals.append(mean)
            dm = f3(np.mean(vals), True)
        cells += [f"{f3(min(nm), True)} to {f3(max(nm), True)}", f"{np.mean(na):.2f}", dm]
    rows.append(PNAME[pipe] + " & " + " & ".join(cells) + r" \\")
open(f"{OUT}/tables/leak.tex", "w").write("\n".join(rows) + "\n")
facts["leak_nperm"] = {"D": L["D"]["n_perm"], "H": None}
facts["leak_null_sets"] = {"D": len(L["D"]["null_datasets"]), "H": len(L["H"]["null_datasets"])}

# ---------------------------------------------------------------- Table 3: threshold vs SMOTE
TV = ["base", "prior", "tuned", "smote_fs", "fs_smote"]
TNAME = {"base": "Threshold 0.5", "prior": "Threshold = training minority share",
         "tuned": "Threshold tuned by inner CV (G-mean)", "smote_fs": r"SMOTE$\to$FS, threshold 0.5",
         "fs_smote": r"FS$\to$SMOTE, threshold 0.5"}
T = {"D": pickle.load(open(f"{D}/results_thr.pkl", "rb")), "H": pickle.load(open(f"{H}/results_thr.pkl", "rb"))}
for s in "DH":
    T[s] = {n: v for n, v in T[s].items() if n not in CEILING}
rows = []
for v in TV:
    cells = []
    for s in "DH":
        for m in ("gmean", "mcc"):
            cells.append(f"{np.mean([[T[s][n][(fs, v, c)][m] for n in T[s]] for fs in 'FH' for c in ('knn', 'lr')]):.3f}")
    rows.append(TNAME[v] + " & " + " & ".join(cells) + r" \\")
# Tomek variants (run_tomek.py: same folds, seed and filters; prior recomputed there and identical)
TK = {"D": pickle.load(open(f"{D}/results_tomek.pkl", "rb")), "H": pickle.load(open(f"{H}/results_tomek.pkl", "rb"))}
KNAME = {"tomek_fs": r"Tomek links$\to$FS, threshold 0.5", "fs_tomek": r"FS$\to$Tomek links, threshold 0.5",
         "smtk_fs": r"SMOTE+Tomek$\to$FS, threshold 0.5", "fs_smtk": r"FS$\to$SMOTE+Tomek, threshold 0.5"}
rows.append(r"\midrule")
for v in KNAME:
    cells = []
    for s in "DH":
        for m in ("gmean", "mcc"):
            cells.append(f"{np.mean([[TK[s][n][(fs, v, c)][m] for n in TK[s]] for fs in 'FH' for c in ('knn', 'lr')]):.3f}")
    rows.append(KNAME[v] + " & " + " & ".join(cells) + r" \\")
open(f"{OUT}/tables/thr.tex", "w").write("\n".join(rows) + "\n")
facts["thr_n"] = {s: len(T[s]) for s in "DH"}

# ---------------------------------------------------------------- Table 4: wrapper vs filter
C = {"D": load(f"{D}/results_cap"), "H": load(f"{H}/results_cap")}
X = {"D": load(f"{D}/results_xclf"), "H": load(f"{H}/results_xclf")}
R = {"D": load(f"{D}/results_ridge"), "H": load(f"{H}/results_ridge")}
WH = {"D": load(f"{D}/results_wrapH"), "H": load(f"{H}/results_wrapH")}


def xp(s, n, m, cls):
    return np.mean([np.mean([X[s][n]["res"][(m, c)][k] for k in ("gmean", "mcc", "aucpr")]) for c in cls])


def rp(s, n, m, cls):
    return np.mean([np.mean([R[s][n]["res"][(m, c)][k] for k in ("gmean", "mcc", "aucpr")]) for c in cls])


def cell(t):
    m, w, l, p = t
    star = r"$^{*}$" if p < 0.05 else ""
    return f"{f3(m, True)}{star} ({w}/{l})"


rows = []
for k in (10, 20, 30, 40):
    cells = []
    for s in "DH":
        names = sorted(C[s])
        cells.append(cell(paired([prim(C[s][n]["res"][f"cap{k}"]) for n in names], [prim(C[s][n]["res"][f"H{k}"]) for n in names])))
        cells.append(cell(paired([xp(s, n, f"cap{k}", ("lr", "svm")) for n in names], [xp(s, n, f"H{k}", ("lr", "svm")) for n in names])))
        cells.append(cell(paired([xp(s, n, f"cap{k}", ("rf",)) for n in names], [xp(s, n, f"H{k}", ("rf",)) for n in names])))
    rows.append(f"Cap {k} $-$ \\Hel{{}}{k} & " + " & ".join(cells) + r" \\")
cells = []
for s in "DH":
    names = sorted(C[s])
    cells.append(cell(paired([prim(WH[s][n]["res"]["bbrier_a99"]) for n in names], [prim(C[s][n]["res"]["H20"]) for n in names])))
    cells.append(cell(paired([xp(s, n, "single", ("lr", "svm")) for n in names], [xp(s, n, "H20", ("lr", "svm")) for n in names])))
    cells.append(cell(paired([xp(s, n, "single", ("rf",)) for n in names], [xp(s, n, "H20", ("rf",)) for n in names])))
rows.append(r"No cap $-$ \Hel{}20 & " + " & ".join(cells) + r" \\")
cells = []
for s in "DH":
    names = sorted(R[s])
    cells.append(cell(paired([rp(s, n, "ridge_cap20", ("rls",)) for n in names], [rp(s, n, "H20", ("rls",)) for n in names])))
    cells.append(cell(paired([rp(s, n, "ridge_cap20", ("lr", "svm")) for n in names], [rp(s, n, "H20", ("lr", "svm")) for n in names])))
    cells.append("--")
rows.append(r"Ridge, cap 20 $-$ \Hel{}20$^{\ddagger}$ & " + " & ".join(cells) + r" \\")
open(f"{OUT}/tables/wrapfilt.tex", "w").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- Table 5: fidelity
FD, FH = J(f"{D}/summary_fid.json"), J(f"{H}/summary_fid.json")
MNAME = {"acc": "Accuracy", "ba": "Balanced accuracy", "gmean": "G-mean", "mcc": "MCC", "f1": "F1", "f2": "F2",
         "kappa": "Cohen's kappa", "fm": "Fowlkes--Mallows", "auc": "AUC-ROC", "aucpr": "AUC-PR",
         "bbrier": "Balanced Brier", "bll": "Balanced log-loss"}
rD = FD["friedman_measures_avg_over_estimators"]["ranks"]; rH = FH["friedman_measures_avg_over_estimators"]["ranks"]
rows = []
for i, m in enumerate(FD["measures"]):
    pm = FD["primary_mean"][i]
    best = max(FD["primary_mean"][j][e] for j in range(12) for e in range(4))
    cs = [f"{v:.3f}" for v in pm]
    rows.append(f"{MNAME[m]} & " + " & ".join(cs) + f" & {rD[m]:.1f} & {rH[m]:.1f} \\\\")
open(f"{OUT}/tables/fid.tex", "w").write("\n".join(rows) + "\n")
facts["fid"] = dict(D_p=FD["friedman_measures_avg_over_estimators"]["p"], H_p=FH["friedman_measures_avg_over_estimators"]["p"],
                    D_best=FD["best_combo"], D_best_vs=FD["best_vs_common"], H_best_vs=FH["best_vs_common"])

# ---------------------------------------------------------------- Table 6: EMT
E = {"D": J(f"{D}/summary_emt.json"), "H": J(f"{H}/summary_emt.json")}
CP = {"D": J(f"{D}/summary_cap.json"), "H": J(f"{H}/summary_cap.json")}
ENAME = [("single", "Single-task BGWO, no cap"), ("cap20", "Single-task BGWO, cap 20"),
         ("emt_f", r"EMT, 1 auxiliary task (ANOVA $F$)"), ("emt_relief", "EMT, 1 auxiliary task (Relief-F)"),
         ("emt_min3", "EMT, 3 minority-aware tasks"), ("emt_min3_rl", r"EMT, 3 minority-aware tasks + UCB1"),
         ("emt_rand3", "EMT, 3 random tasks (control)"), ("H-top20", "Hellinger top-20 filter")]
rows = []
for m, lab in ENAME:
    cells = []
    for s in "DH":
        if m == "cap20":
            pr, st = CP[s]["primary"]["cap20"], CP[s]["stab"]["cap20"]
        else:
            pr, st = E[s]["mean_primary"][m], E[s]["mean_stability"][m]
        cells += [f"{pr:.3f}", f"{st:.2f}"]
    rows.append(lab + " & " + " & ".join(cells) + r" \\")
open(f"{OUT}/tables/emt.tex", "w").write("\n".join(rows) + "\n")
facts["bandit_share"] = {s: E[s]["bandit_pick_share_(hell,auc,bfisher)"] for s in "DH"}
for s in "DH":
    names = sorted(C[s])
    Em = load(f"{[D, H]['DH'.index(s)]}/results_emt")
    facts[f"emt_vs_cap20_{s}"] = paired([prim(Em[n]["res"]["emt_min3"]) for n in names], [prim(C[s][n]["res"]["cap20"]) for n in names])
    facts[f"emt_vs_rand_{s}"] = paired([prim(Em[n]["res"]["emt_min3"]) for n in names], [prim(Em[n]["res"]["emt_rand3"]) for n in names])
    facts[f"rl_vs_unif_{s}"] = paired([prim(Em[n]["res"]["emt_min3_rl"]) for n in names], [prim(Em[n]["res"]["emt_min3"]) for n in names])

# ---------------------------------------------------------------- Table: additional baselines (k = 20)
SB = J(f"{D}/summary_base.json")
BNAME = [("H", "Hellinger"), ("F", "ANOVA $F$"), ("L1", "L1-logistic"), ("EN", "Elastic net"), ("mRMR", "mRMR"),
         ("RFimp", "RF impurity importance"), ("RFperm", "RF permutation importance"), ("wrap", "Wrapper, cap 20")]
GR = ("knn", "lrsvm", "rf")
rows = []
for m, lab in BNAME:
    cells = [f"{SB['mean'][s][f'{m}|20|{cl}']:.3f}" for s in ("disc", "held") for cl in GR]
    for cl in GR:
        if m == "H":
            cells.append("--"); continue
        r = SB["vsH"][f"{m}|20|{cl}"]["both|dataset"]
        mark = r"$^{*}$" if r["p"] < 0.05 else ""
        eq = r"$^{\approx}$" if r["equivalent"] else ""
        cells.append(f"{f3(r['mean'], True)}{mark}{eq}")
    rows.append(lab + " & " + " & ".join(cells) + r" \\")
# rows added in review (run_extra.py, same folds): SVM-RFE, cost-sensitive SVM-RFE, Hellinger with F tie-break, no selection
SX = J(f"{D}/summary_extra.json")
XNAME = [("RFE", 20, "SVM-RFE"), ("RFEw", 20, "SVM-RFE, class-weighted"), ("Hf", 20, r"Hellinger, $F$ tie-break"),
         ("ALL", 0, "All features (no selection)")]
xrows = []
for m, k, lab in XNAME:
    cells = [f"{SX['extra_mean'][s][f'{m}|{k}|{cl}']:.3f}" for s in ("disc", "held") for cl in GR]
    for cl in GR:
        r = SX["extra_vsH"][f"{m}|20|{cl}"]["both|dataset"]
        mark = r"$^{*}$" if r["p"] < 0.05 else ""
        eq = r"$^{\approx}$" if r["equivalent"] else ""
        cells.append(f"{f3(r['mean'], True)}{mark}{eq}")
    xrows.append(lab + " & " + " & ".join(cells) + r" \\")
rows = rows[:1] + [xrows[2]] + rows[1:-1] + [xrows[0], xrows[1], xrows[3], r"\midrule", rows[-1]]
open(f"{OUT}/tables/base.tex", "w").write("\n".join(rows) + "\n")
facts["extra"] = {k: SX[k] for k in ("extra_n", "ties", "extra_stab", "extra_time", "tie_worst")}
facts["base"] = {k: SB[k] for k in ("n", "stab", "time")}

# ---------------------------------------------------------------- Table: MF-CSO (re-implemented), repeats 0-1
SMF = J(f"{D}/summary_mfcso.json")
MFN = [("mfcso", "MF-CSO (re-implemented)"), ("H_same_k", "Hellinger, same size as MF-CSO"), ("H20", "Hellinger top-20"), ("cap20", "Wrapper, cap 20")]
rows = []
for m, lab in MFN:
    cells = [f"{SMF['mean'][s][f'{m}|{cl}']:.3f}" for s in ("disc", "held") for cl in GR]
    if m == "mfcso":
        cells += ["--"] * 3
    else:
        for cl in GR:
            r = SMF["contrasts"][f"mfcso - {m}|{cl}"]["both|family"]
            mark = r"$^{*}$" if r["p"] < 0.05 else ""
            cells.append(f"{f3(r['mean'], True)}{mark}")
    rows.append(lab + " & " + " & ".join(cells) + r" \\")
open(f"{OUT}/tables/mfcso.tex", "w").write("\n".join(rows) + "\n")
facts["mfcso"] = {k: SMF[k] for k in ("n", "size", "stab", "time")}

# ---------------------------------------------------------------- Table: sssHD (re-implemented), repeats 0-1
if os.path.exists(f"{D}/summary_ssshd.json"):
    SS = J(f"{D}/summary_ssshd.json")
    rows = []
    for k in (10, 20, 30, 40):
        cells = []
        for s_ in ("disc", "held"):
            for cl in GR:
                r = SS["vsH"][f"{k}|{cl}"][f"{s_}|dataset"]
                cells.append(f"{f3(r['mean'], True)}{'$^{*}$' if r['p'] < 0.05 else ''}")
        for cl in GR:
            r = SS["vsH"][f"{k}|{cl}"]["both|family"]
            mark = r"$^{*}$" if r["p"] < 0.05 else (r"$^{\approx}$" if r["equivalent"] else "")
            cells.append(f"{f3(r['mean'], True)}{mark} ({r['wins']}/{r['losses']})")
        rows.append(f"{k} & " + " & ".join(cells) + r" \\")
    open(f"{OUT}/tables/ssshd.tex", "w").write("\n".join(rows) + "\n")
    facts["ssshd"] = {k: SS[k] for k in ("n", "stab", "time", "sel")}

# ---------------------------------------------------------------- Table: further wrapper checks (run_wrap2.py)
if SX.get("wrap2_n", {}).get("disc", 0) > 0:
    W2N = [("bgwo_cap20-H20", "BGWO cap 20 $-$ \\Hel{}20"),
           ("ga_cap20-H20", "GA cap 20 $-$ \\Hel{}20"),
           ("ga_cap20-bgwo_cap20", "GA cap 20 $-$ BGWO cap 20"),
           ("ga_nocap-bgwo_nocap", "GA no cap $-$ BGWO no cap"),
           ("bgwo_cap20_x4-bgwo_cap20", "BGWO cap 20: $4\\times$ budget $-$ $1\\times$$^{\\dagger}$"),
           ("bgwo_nocap_x4-bgwo_nocap", "BGWO no cap: $4\\times$ budget $-$ $1\\times$$^{\\dagger}$"),
           ("bgwo_cap20_x4-H20", "BGWO cap 20, $4\\times$ budget $-$ \\Hel{}20$^{\\dagger}$"),
           ("bgwoF_cap20-F20", "BGWO cap 20, $F$ pool $-$ $F$ top-20"),
           ("bgwoF_cap20-H20", "BGWO cap 20, $F$ pool $-$ \\Hel{}20")]
    rows = []
    for key, lab in W2N:
        cells = []
        for s_ in ("disc", "held"):
            for cl in GR:
                r = SX["wrap2"][f"{key}|{cl}"].get(f"{s_}|dataset")
                cells.append("--" if r is None else f"{f3(r['mean'], True)}{'$^{*}$' if r['p'] < 0.05 else ''}")
        for cl in GR:
            r = SX["wrap2"][f"{key}|{cl}"].get("both|family")
            if r is None:
                cells.append("--"); continue
            mark = r"$^{*}$" if r["p"] < 0.05 else (r"$^{\approx}$" if r["equivalent"] else "")
            cells.append(f"{f3(r['mean'], True)}{mark} ({r['wins']}/{r['losses']})")
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    open(f"{OUT}/tables/wrap2.tex", "w").write("\n".join(rows) + "\n")
    facts["wrap2"] = {k: SX[k] for k in ("wrap2_n", "wrap2_size", "wrap2_stab", "seed_stab", "wrap2_time", "wrap2_mean")}

# ---------------------------------------------------------------- Table: computational cost (if measured)
TP = "/home/claude/hdi/time_methods.json"
if os.path.exists(TP):
    TM = J(TP)
    sets = list(TM)
    TNAME2 = [("filter/embedded:H", "Hellinger"), ("filter/embedded:F", "ANOVA $F$"), ("filter/embedded:L1", "L1-logistic"),
              ("filter/embedded:EN", "Elastic net"), ("filter/embedded:mRMR", "mRMR"),
              ("filter/embedded:RFimp", "RF impurity importance"), ("filter/embedded:RFperm", "RF permutation importance"),
              ("filter/embedded:sssHD", "sssHD (re-implemented)"),
              ("wrapper:BGWO 5-NN cap 20", "Wrapper, 5-NN fitness, cap 20"), ("wrapper:BGWO 5-NN no cap", "Wrapper, 5-NN fitness, no cap"),
              ("wrapper:BGWO ridge cap 20", "Wrapper, ridge fitness, cap 20"), ("wrapper:EMT 3 tasks", "EMT, 3 minority-aware tasks")]
    fmt = lambda v: f"{v:.3f}" if v < 0.01 else (f"{v:.2f}" if v < 10 else f"{v:.0f}")
    rows = [lab + " & " + " & ".join(fmt(TM[s]["times"][k]) for s in sets) + r" \\" for k, lab in TNAME2 if k in TM[sets[0]]["times"]]
    head = " & ".join(f"{s.replace('_', r'\_')}" for s in sets)
    sub = " & ".join(f"{TM[s]['n_train']}$\\times${TM[s]['p']}" for s in sets)
    open(f"{OUT}/tables/time_head.tex", "w").write(f"Method & {head} \\\\\n & {sub} \\\\\n")
    open(f"{OUT}/tables/time.tex", "w").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- Figure 1: forest plot
DEP = J(f"{D}/summary_dep.json")
LAB = {"cap20 - H20 (5-NN final)": "Wrapper cap 20 $-$ Hellinger top-20\n(5-NN, the fitness classifier)",
       "cap20 - H20 (LR/SVM final)": "Wrapper cap 20 $-$ Hellinger top-20\n(LR and SVM)",
       "ridge_cap20 - H20 (ridge/LR/SVM)": "Ridge wrapper cap 20 $-$ Hellinger\ntop-20 (ridge, LR, SVM)",
       "emt_min3 - cap20 (5-NN)": "EMT (3 minority-aware tasks)\n$-$ wrapper cap 20 (5-NN)",
       "bbrier - acc wrapper (5-NN, H pool)": "Balanced-Brier fitness $-$\naccuracy fitness (5-NN)",
       "SMOTE->FS - prior, G-mean": "SMOTE$\\to$FS $-$ prior threshold\n(G-mean)",
       "SMOTE->FS - prior, MCC": "SMOTE$\\to$FS $-$ prior threshold\n(MCC)",
       "FS->SMOTE - prior, G-mean": "FS$\\to$SMOTE $-$ prior threshold\n(G-mean)",
       "FS->SMOTE - prior, MCC": "FS$\\to$SMOTE $-$ prior threshold\n(MCC)"}
SER = [("disc|dataset", "Discovery, 34 datasets", "#2a78d6", "o"), ("held|dataset", "Held-out, 18 datasets", "#eb6834", "s"),
       ("both|family", "Pooled, 38 families", "#1baf7a", "D")]
keys = list(LAB)
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.edgecolor": "#52514e", "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(6.3, 6.2))
ax.axvspan(-0.02, 0.02, color="#f0efec", zorder=0, lw=0)
ax.axvline(0, color="#52514e", lw=0.8, zorder=1)
for i, k in enumerate(keys):
    for j, (sk, lab, col, mk) in enumerate(SER):
        r = DEP[k][sk]; yv = -(i * 4 + j)
        ax.plot(r["ci90"], [yv, yv], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(r["mean"], yv, mk, ms=6, color=col, mec="#fcfcfb", mew=1, zorder=3, label=lab if i == 0 else None)
ax.set_yticks([-(i * 4 + 1) for i in range(len(keys))]); ax.set_yticklabels([LAB[k] for k in keys])
ax.set_xlabel("Paired difference (mean and 90% bootstrap interval)")
ax.grid(axis="x", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.0)
ax.text(0, 1.5, "equivalence band $\\pm$0.02", ha="center", va="bottom", fontsize=7, color="#52514e")
ax.set_ylim(-(len(keys) * 4 - 1.5), 2.6)
fig.tight_layout(); fig.savefig(f"{OUT}/figs/forest.pdf", bbox_inches="tight"); plt.close(fig)
facts["dep"] = DEP

# ---------------------------------------------------------------- Figure 2: stability
SM = [("single", "BGWO\nno cap"), ("cap10", "BGWO\ncap 10"), ("cap20", "BGWO\ncap 20"), ("cap40", "BGWO\ncap 40"),
      ("emt_min3", "EMT\n3 tasks"), ("H10", "Hellinger\ntop-10"), ("H20", "Hellinger\ntop-20"), ("H40", "Hellinger\ntop-40")]
val = {}
for s in "DH":
    val[s] = [E[s]["mean_stability"][m] if m.startswith("emt") else CP[s]["stab"][m] for m, _ in SM]
fig, ax = plt.subplots(figsize=(6.3, 2.6))
xs = np.arange(len(SM)); w = 0.36
for off, s, col, lab in ((-w / 2 - 0.01, "D", "#2a78d6", "Discovery (34)"), (w / 2 + 0.01, "H", "#eb6834", "Held-out (18)")):
    b = ax.bar(xs + off, val[s], w, color=col, label=lab, zorder=2)
    for x, v in zip(xs + off, val[s]):
        ax.text(x, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color="#0b0b0b")
ax.set_xticks(xs); ax.set_xticklabels([l for _, l in SM], fontsize=7.5)
ax.set_ylabel("Nogueira stability"); ax.set_ylim(0, 0.56)
ax.axvline(4.5, color="#c3c2b7", lw=0.6, ls=(0, (3, 3)))
ax.grid(axis="y", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/figs/stability.pdf"); plt.close(fig)
facts["stab"] = {s: dict(zip([m for m, _ in SM], val[s])) for s in "DH"}

json.dump(facts, open(f"{OUT}/facts.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in facts.items() if k != "dep"}, indent=1, default=float))

# ---------------------------------------------------------------- Supplementary tables from the review analyses
if os.path.exists(f"{D}/summary_review.json"):
    RV = J(f"{D}/summary_review.json")
    MN = [("F", "ANOVA $F$"), ("L1", "L1-logistic"), ("EN", "Elastic net"), ("mRMR", "mRMR"),
          ("RFimp", "RF impurity"), ("RFperm", "RF permutation"), ("cap20", "Wrapper, cap 20")]
    G3 = ("knn", "lrsvm", "rf")
    def mk(r):
        return f"{f3(r['mean'], True)}{'$^{*}$' if r['p'] < 0.05 else ''}"
    rows = []
    for m, lab in MN:
        cells = []
        for q in ("gmean", "mcc", "aucpr"):
            for g in G3:
                key = f"cap20-H20|{q}|{g}" if m == "cap20" else f"{m}-H|{q}|{g}"
                cells.append(mk(RV["components"][key]))
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    open(f"{OUT}/tables/supp_comp.tex", "w").write("\n".join(rows) + "\n")
    rows = []
    for m, lab in MN:
        cells = []
        for g in G3:
            key = f"cap20-H20|{g}" if m == "cap20" else f"H-{m}|{g}"
            r = RV["ir_moderation"][key]
            sign = -1 if m == "cap20" else 1      # report every row as (Hellinger advantage) except the wrapper row
            cells.append(f"{f3(r['spearman_all'], True)} ({r['p_spearman']:.2f}); {r['n_pos']}/{r['n_neg']}")
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    open(f"{OUT}/tables/supp_ir.tex", "w").write("\n".join(rows) + "\n")
    rows = []
    for k in (10, 20, 30, 40):
        for m, lab in MN[:-1]:
            cells = [f3(RV["text_tasks"]["diff"][f"{m}|{k}|{g}"]["mean"], True) for g in G3]
            rows.append((f"{k}" if m == "F" else "") + f" & {lab} & " + " & ".join(cells) + r" \\")
        if k != 40:
            rows.append(r"\midrule")
    open(f"{OUT}/tables/supp_text.tex", "w").write("\n".join(rows) + "\n")
    facts["review"] = {"sizes": RV["sizes"], "threshold_auc": RV["threshold_auc"]}
    json.dump(facts, open(f"{OUT}/facts.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- cell shading of the comparison tables
# Each numeric cell of tables/base.tex and tables/wrap2.tex is shaded by its difference from the reference
# (Hellinger filter): blue = better, red = worse, white for |d| < 0.005, full tint at |d| >= 0.05.
# Shading repeats the numbers; the * and ~ marks are kept so the tables also read in grey scale.
import re as _re
_NUM = _re.compile(r"^\s*(\$-\$|\+|-)?(\d+\.\d+)")
POS, NEG, TINT, SAT = (0.13, 0.40, 0.67), (0.80, 0.18, 0.15), 0.45, 0.05


def _val(c):
    m = _NUM.match(c)
    if not m:
        return None
    return (-1 if m.group(1) in ("$-$", "-") else 1) * float(m.group(2))


def _shade(c, d):
    if d is None or abs(d) < 0.005:
        return c
    a = TINT * min(abs(d) / SAT, 1.0)
    rgb = ",".join(f"{1 - a * (1 - x):.3f}" for x in (POS if d > 0 else NEG))
    return r"\cellcolor[rgb]{" + rgb + "}" + c.strip() + " "


def _rows(path):
    return open(path).read().rstrip("\n").split("\n")


def _split(row):
    return [c for c in row.rstrip().removesuffix(r"\\").split("&")]


# wrapper row of Table base: its own Hellinger top-20 filter (results_cap / results_xclf), as in the caption
def _own_h20():
    out = []
    for d in (D, H):
        Cp, Xc = load(f"{d}/results_cap"), load(f"{d}/results_xclf")
        names = sorted(Cp)
        for cl in (("knn",), ("lr", "svm"), ("rf",)):
            out.append(float(np.mean([np.mean([prim(Cp[n]["res"]["H20"]) if c == "knn" else prim(Xc[n]["res"][("H20", c)])
                                               for c in cl]) for n in names])))
    return out


rows = _rows(f"{OUT}/tables/base.tex")
ref = [_val(c) for c in _split(rows[0])[1:7]]
own = _own_h20()
out = []
for r in rows:
    if "&" not in r or r.startswith("Hellinger &"):
        out.append(r); continue
    c = _split(r)
    base = own if c[0].startswith("Wrapper") else ref
    c = [c[0]] + [_shade(x, None if _val(x) is None else _val(x) - base[i]) for i, x in enumerate(c[1:7])] + \
        [_shade(x, _val(x)) for x in c[7:]]
    out.append(" & ".join(x.strip() for x in c) + r" \\")
open(f"{OUT}/tables/base.tex", "w").write("\n".join(out) + "\n")
facts["own_h20"] = own

if os.path.exists(f"{OUT}/tables/wrap2.tex"):
    out = []
    for r in _rows(f"{OUT}/tables/wrap2.tex"):
        if "&" not in r:
            out.append(r); continue
        c = _split(r)
        out.append(" & ".join([c[0].strip()] + [_shade(x, _val(x)).strip() for x in c[1:]]) + r" \\")
    open(f"{OUT}/tables/wrap2.tex", "w").write("\n".join(out) + "\n")
json.dump(facts, open(f"{OUT}/facts.json", "w"), indent=1, default=float)
