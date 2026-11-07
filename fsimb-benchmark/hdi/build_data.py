"""Phase 1: build the benchmark suite (data/v2/*.npz + inventory.csv).

Three kinds of dataset:
  natural    - a binary task from a public microarray / omics set, with its
               own (natural) imbalance; multiclass sets become
               one-class-vs-rest, with the named class as minority (y = 1).
  controlled - minority subsampled from a large, near-balanced set so that
               IR is exactly 2, 5, 10 or 20 with the majority size held fixed.
  synthetic  - make_classification with 20 known relevant features
               (10 informative + 10 redundant), columns shuffled.

Only label-free cleaning is done here (drop constant columns); everything
that looks at labels happens inside the outer CV folds later.

Usage: python -I build_data.py <raw_dir> <out_dir>
"""
import sys, csv, warnings
import numpy as np, rdata, scipy.io as sio
from sklearn.datasets import make_classification

warnings.filterwarnings("ignore")
RAW, OUT = sys.argv[1], sys.argv[2]
rows = []


def save(name, X, y, kind, source, task, relevant=None):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    keep = X.std(0) > 0
    if relevant is not None:
        # map relevant column ids through the constant-column filter
        new_idx = -np.ones(X.shape[1], int)
        new_idx[keep] = np.arange(keep.sum())
        relevant = new_idx[relevant]
        relevant = relevant[relevant >= 0]
    X = X[:, keep]
    assert not np.isnan(X).any(), name
    m = int(y.sum())
    extra = {"relevant": relevant} if relevant is not None else {}
    np.savez_compressed(f"{OUT}/{name}.npz", X=X, y=y, **extra)
    rows.append(dict(name=name, kind=kind, source=source, task=task, n=len(y), p=X.shape[1],
                     minority=m, majority=len(y) - m, IR=round((len(y) - m) / m, 2)))
    print(f"{name:22s} n={len(y):4d} p={X.shape[1]:6d} min={m:3d} IR={(len(y)-m)/m:5.2f}")


def dm(name):
    o = rdata.read_rda(f"{RAW}/{name}.RData")[name]
    return np.asarray(o["x"], float), np.asarray(o["y"]).astype(str)


def skf(name):
    m = sio.loadmat(f"{RAW}/skf/{name}.mat")
    X = m["X"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, float)
    return X, np.asarray(m["Y"]).ravel()


# ---------------- natural imbalance (minority label in task) ----------------
DM_TASKS = {  # name: minority label (rest = majority)
    "alon": "n", "golub": "AML", "gravier": "poor", "shipp": "FL", "tian": "MRI-no-lytic-lesion sample",
    "gordon": "mesothelioma", "chin": "negative", "pomeroy": "died", "subramanian": "WT",
    "burczynski": "ulcerative colitis", "sorlie": "2", "christensen": "placenta",
    "chiaretti": "ALL1/AF4", "yeoh": "BCR",
}
for name, lab in DM_TASKS.items():
    X, y = dm(name)
    save(name, X, (y == lab).astype(int), "natural", "datamicroarray", f"{lab} vs rest")

SKF_TASKS = {"GLI_85": 1, "TOX_171": 3, "Carcinom": 1, "lymphoma": 2, "CLL_SUB_111": 1, "lung": 2}
for name, lab in SKF_TASKS.items():
    X, y = skf(name)
    save(name, X, (y == lab).astype(int), "natural", "scikit-feature", f"class {lab} vs rest")

# ---------------- controlled IR variants ----------------
rng = np.random.default_rng(2026)
CTRL = {  # name: (positive label = minority, majority size, IR levels)
    "gisette": (1, 200, [2, 5, 10, 20]),
    "BASEHOCK": (2, 200, [2, 5, 10, 20]),
    "SMK_CAN_187": (2, None, [2, 5, 10]),
    "arcene": (1, None, [2, 5, 10]),
}
for name, (pos, nmaj, irs) in CTRL.items():
    X, y = skf(name)
    yb = (y == pos).astype(int)
    mi, ma = np.where(yb == 1)[0], np.where(yb == 0)[0]
    ma = np.sort(rng.choice(ma, size=nmaj, replace=False)) if nmaj else ma
    order = rng.permutation(mi)            # nested: IR 20 minority is a subset of IR 10, etc.
    for ir in irs:
        k = int(round(len(ma) / ir))
        idx = np.sort(np.concatenate([ma, order[:k]]))
        save(f"{name}_ir{ir}", X[idx], yb[idx], "controlled", "scikit-feature",
             f"class {pos} subsampled to IR {ir}")

# ---------------- synthetic, known relevant features ----------------
SYN = [("syn_p2000_s15_ir10", 330, 2000, 1.5, 10), ("syn_p5000_s15_ir10", 330, 5000, 1.5, 10),
       ("syn_p2000_s20_ir20", 420, 2000, 2.0, 20), ("syn_p5000_s20_ir5", 180, 5000, 2.0, 5)]
for i, (name, n, p, sep, ir) in enumerate(SYN):
    w = ir / (ir + 1.0)
    X, y = make_classification(n_samples=n, n_features=p, n_informative=10, n_redundant=10,
                               n_repeated=0, n_clusters_per_class=2, weights=[w], flip_y=0.0,
                               class_sep=sep, shuffle=False, random_state=100 + i)
    perm = np.random.default_rng(100 + i).permutation(p)
    X = X[:, perm]
    save(name, X, (y == 1).astype(int), "synthetic", "make_classification",
         f"class_sep {sep}, 20 relevant features", relevant=np.where(perm < 20)[0])

with open(f"{OUT}/inventory.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "datasets")
