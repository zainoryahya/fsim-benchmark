"""Main experiment: 7 methods x 10 datasets, 5-fold x 4 repeats nested evaluation."""
import pickle, time, numpy as np
from joblib import Parallel, delayed
from fsimb.pipeline import run_dataset

METHODS = ["F-top20", "H-top20", "GWO-Acc", "GWO-Gm", "EMT-Relief", "EMT-MinAware", "EMT-MinAware-RL"]
DATASETS = ["alon", "golub", "gravier", "shipp", "tian", "gordon",
            "chowdary_ir5", "singh_ir5", "synth_p2000", "synth_p5000"]

def one(name):
    d = np.load(f"data/{name}.npz")
    t = time.time()
    res = run_dataset(name, d["X"], d["y"], METHODS, n_splits=5, n_repeats=4, seed=2026)
    meta = {"n": d["X"].shape[0], "p": d["X"].shape[1], "minority": int(d["y"].sum()),
            "relevant": d["relevant"] if "relevant" in d.files else None, "secs": time.time() - t}
    print(name, f"{meta['secs']:.0f}s", flush=True)
    return name, res, meta

if __name__ == "__main__":
    out = Parallel(n_jobs=2)(delayed(one)(n) for n in DATASETS)
    with open("results_main.pkl", "wb") as f:
        pickle.dump({"methods": METHODS, "results": {n: (r, m) for n, r, m in out}}, f)
    print("saved")
