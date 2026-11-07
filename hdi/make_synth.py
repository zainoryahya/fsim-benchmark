import numpy as np
from sklearn.datasets import make_classification
for name, n, p, w, seed in [("synth_p2000", 200, 2000, 0.9, 1), ("synth_p5000", 150, 5000, 0.88, 2)]:
    X, y = make_classification(n_samples=n, n_features=p, n_informative=10, n_redundant=10,
        n_repeated=0, n_clusters_per_class=2, weights=[w], flip_y=0.0, class_sep=1.0,
        shuffle=False, random_state=seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(p)            # hide the relevant columns
    X = X[:, perm]
    relevant = np.where(perm < 20)[0]    # new positions of the 20 informative+redundant
    y = (y == 1).astype(int)
    np.savez_compressed(f"data/{name}.npz", X=X, y=y, relevant=relevant)
    print(name, X.shape, "minority", y.sum(), "IR", round((len(y)-y.sum())/y.sum(),2))
