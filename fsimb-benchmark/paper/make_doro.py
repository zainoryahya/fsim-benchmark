"""Dorothea table (tables/doro.tex) from hdi_ho/summary_doro.json."""
import json, os
S = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hdi_ho", "summary_doro.json")))
f = lambda x: f"{x:+.3f}".replace("-", "$-$")
L = S["leak"]
rng = lambda pipes, k: (min(L[p][c][k] for p in pipes for c in L[p]), max(L[p][c][k] for p in pipes for c in L[p]))
a, b = rng(["FS only", "SMOTE->FS", "FS->SMOTE"], "null_mcc")
c, d = rng(["leak SMOTE-first"], "null_mcc")
e, g = rng(["leak FS-first"], "null_mcc"); ea, ga = rng(["leak FS-first"], "null_auc")
nfs = sum(L["leak FS-first"][k]["null_mcc"] > 0.10 for k in L["leak FS-first"])
st = S["stability"]
capst = [st[f"cap{k}"] for k in (10, 20, 30, 40)]; emtst = [st[m] for m in ("emt_f", "emt_relief", "emt_min3", "emt_min3_rl")]
hst = [st[f"H{k}"] for k in (10, 20, 30, 40)]
cl = S["cap_minus_H_lrsvm"]
rows = [
 (r"P1a In-fold null MCC in $[-0.03, +0.03]$", f"{f(a)} to {f(b)}", "Met"),
 (r"P1b SMOTE-before-CV null MCC $> +0.10$", f"{f(c)} to {f(d)}", "Met"),
 (r"P1c Filter-before-CV null MCC $> +0.10$", f"{f(e)} to {f(g)} (null AUC {ea:.2f}--{ga:.2f})", f"Met in {nfs} of 4 combinations; leak visible in AUC"),
 (r"P2 $|$SMOTE $-$ prior threshold$|$ in G-mean $\le 0.02$", f"SMOTE$\\to$FS {f(S['P2_smote_fs_minus_prior_gmean'])}; FS$\\to$SMOTE {f(S['P2_fs_smote_minus_prior_gmean'])}", r"Met for FS$\to$SMOTE; SMOTE$\to$FS just outside (worse than the threshold)"),
 (r"P3 Cap 20 $-$ \Hel{}20 on LR/SVM $< +0.02$", f"{f(S['P3_cap20_minus_H20_lrsvm'])} (caps 10/30/40: {f(cl['10'])}, {f(cl['30'])}, {f(cl['40'])})", "Met; at caps 30--40 (not registered) the wrapper was ahead"),
 (r"P4 EMT (3 tasks) $-$ cap 20 (5-NN) $\le +0.02$", f(S["P4_emt_min3_minus_cap20"]), "Met"),
 (r"P5 Wrapper stability $<$ filter stability", f"caps {min(capst):.2f}--{max(capst):.2f}, EMT {min(emtst):.2f}--{max(emtst):.2f}, filter {min(hst):.2f}--{max(hst):.2f}", "Met"),
]
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables", "doro.tex"), "w").write("\n".join(f"{r[0]} & {r[1]} & {r[2]} \\\\" for r in rows) + "\n")
print(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables", "doro.tex")).read())
