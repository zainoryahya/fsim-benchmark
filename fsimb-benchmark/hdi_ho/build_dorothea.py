"""Dorothea (NIPS 2003): train+valid merged, label-free filter (feature active in >= 1% of samples)."""
import sys, csv, numpy as np
d, out = sys.argv[1], sys.argv[2]
def rows(f): return [np.array(l.split(), int) - 1 for l in open(f)]
R = rows(f"{d}/ea723fdf-dorothea_train.data") + rows(f"{d}/1868a739-dorothea_valid.data")
y = np.r_[np.loadtxt(f"{d}/d8bcba24-dorothea_train.labels"), np.loadtxt(f"{d}/4dd8b1bb-dorothea_valid.labels")]
y = (y == 1).astype(np.int8)
X = np.zeros((len(R), 100000), np.float32)
for i, r in enumerate(R): X[i, r] = 1.0
keep = X.sum(0) >= np.ceil(0.01 * len(R))
X = X[:, keep]
np.savez_compressed(f"{out}/dorothea.npz", X=X, y=y)
m = int(y.sum())
with open(f"{out}/inventory.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["name", "kind", "source", "task", "n", "p", "minority", "majority", "IR"])
    w.writerow(["dorothea", "natural", "NIPS2003", "active vs inactive", len(y), X.shape[1], m, len(y) - m, round((len(y) - m) / m, 2)])
print("dorothea n", len(y), "p", X.shape[1], "min", m, "IR", round((len(y) - m) / m, 2))
