import numpy as np
# controlled-imbalance variants: keep all majority, subsample minority to IR ~ 5
for src, ir in [("chowdary", 5), ("singh", 5)]:
    d = np.load(f"data/{src}.npz"); X, y = d["X"], d["y"]
    rng = np.random.default_rng(42)
    mi, ma = np.where(y == 1)[0], np.where(y == 0)[0]
    keep_min = rng.choice(mi, size=max(8, len(ma) // ir), replace=False)
    idx = np.sort(np.concatenate([ma, keep_min]))
    np.savez_compressed(f"data/{src}_ir{ir}.npz", X=X[idx], y=y[idx])
    print(src, "->", len(idx), "minority", len(keep_min), "IR", round(len(ma)/len(keep_min), 2))
