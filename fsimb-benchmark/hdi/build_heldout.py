"""Held-out validation suite, fixed BEFORE any model was run on it.

Rules (declared in the project document before the runs):
  * natural binary sets: minority = the smaller class (chowdary: colon, singh: Normal, west: positive,
    borovecki: control);
  * multiclass sets: minority = the smallest class with at least 10 samples (khan: NB, nakayama: synovial sarcoma,
    su: class 4, sun: non-tumor); for nakayama the rule was misapplied: two classes
    with 15 samples (dedifferentiated liposarcoma, myxofibrosarcoma) are smaller than synovial sarcoma (16), which
    was used (disclosed in the manuscript); nci9 has no class with >= 10 samples and is skipped;
  * 10-class image sets (orlraws10P, pixraw10P, warpAR10P, warpPIE10P): class 1 versus the rest (IR 9);
  * controlled IR: singh (Tumor subsampled), chowdary (colon subsampled) at IR 5; PCMAC and RELATHE
    (majority 200, minority class 2 subsampled) at IR 5 and 10.
Only label-free cleaning (constant columns dropped). Usage: python build_heldout.py <raw_dir> <out_dir>
"""
import sys, csv, warnings
import numpy as np, rdata, scipy.io as sio
warnings.filterwarnings("ignore")
RAW, OUT = sys.argv[1], sys.argv[2]
rows = []


def save(name, X, y, kind, source, task):
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int8)
    X = X[:, X.std(0) > 0]
    assert not np.isnan(X).any(), name
    m = int(y.sum())
    np.savez_compressed(f"{OUT}/{name}.npz", X=X, y=y)
    rows.append(dict(name=name, kind=kind, source=source, task=task, n=len(y), p=X.shape[1],
                     minority=m, majority=len(y) - m, IR=round((len(y) - m) / m, 2)))
    print(f"{name:20s} n={len(y):4d} p={X.shape[1]:6d} min={m:3d} IR={(len(y)-m)/m:5.2f}")


def dm(n):
    o = rdata.read_rda(f"{RAW}/{n}.RData")[n]
    return np.asarray(o["x"], float), np.asarray(o["y"]).astype(str)


def skf(n):
    m = sio.loadmat(f"{RAW}/skf/{n}.mat"); X = m["X"]
    return (X.toarray() if hasattr(X, "toarray") else np.asarray(X, float)), np.asarray(m["Y"]).ravel()


for n, lab in {"chowdary": "colon", "singh": "Normal", "west": "positive", "borovecki": "control",
               "khan": "NB", "nakayama": "synovial sarcoma", "su": "4", "sun": "non-tumor"}.items():
    X, y = dm(n)
    save(n, X, (y == lab).astype(int), "natural", "datamicroarray", f"{lab} vs rest")
for n in ["orlraws10P", "pixraw10P", "warpAR10P", "warpPIE10P"]:
    X, y = skf(n)
    save(n, X, (y == 1).astype(int), "natural", "scikit-feature", "class 1 vs rest")

rng = np.random.default_rng(3030)


def controlled(name, X, yb, ir, nmaj=None):
    mi, ma = np.where(yb == 1)[0], np.where(yb == 0)[0]
    if nmaj:
        ma = np.sort(rng.choice(ma, size=nmaj, replace=False))
    k = int(round(len(ma) / ir))
    idx = np.sort(np.concatenate([ma, rng.permutation(mi)[:k]]))
    save(f"{name}_ir{ir}", X[idx], yb[idx], "controlled", "mixed", f"subsampled to IR {ir}")


X, y = dm("singh");    controlled("singh", X, (y == "Tumor").astype(int), 5)
X, y = dm("chowdary"); controlled("chowdary", X, (y == "colon").astype(int), 5)
for n in ["PCMAC", "RELATHE"]:
    X, y = skf(n)
    for ir in (5, 10):
        controlled(n, X, (y == 2).astype(int), ir, nmaj=200)
with open(f"{OUT}/inventory.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "datasets")
