"""Sanity baseline on the v2 suite: filter top-20 + 3-NN, nested 5-fold x 2."""
import csv, json, numpy as np
from joblib import Parallel, delayed
from fsimb.pipeline import run_dataset
inv = list(csv.DictReader(open("data/v2/inventory.csv")))
def one(r):
    d = np.load(f"data/v2/{r['name']}.npz")
    X, y = d["X"].astype(float), d["y"].astype(int)
    out = run_dataset(r["name"], X, y, ["F-top20", "H-top20"], n_splits=5, n_repeats=2, seed=11)
    res = {m: round(float(np.mean(out[m]["gmean"])), 3) for m in out}
    if "relevant" in d.files:
        rel = set(d["relevant"].tolist())
        res["recovery_F"] = round(float(np.mean([len(rel & set(s.tolist())) / len(s) for s in out["F-top20"]["subsets"]])), 3)
        res["n_relevant"] = len(rel)
    return r["name"], res
out = dict(Parallel(n_jobs=2)(delayed(one)(r) for r in inv))
json.dump(out, open("data/v2/baseline_filter.json", "w"), indent=1)
for k, v in out.items(): print(f"{k:22s}", v)
