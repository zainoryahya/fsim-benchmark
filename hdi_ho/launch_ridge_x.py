"""Launcher: runs run_ridge.run_one in parallel from a normal module (avoids pickling __main__)."""
import sys, os, glob
from joblib import Parallel, delayed
import run_ridge_x as R

if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    paths = [p for p in sorted(glob.glob(f"{data_dir}/*.npz")) if os.path.basename(p)[:-4] not in R.CEILING
             and not os.path.exists(f"{out_dir}/{os.path.basename(p)[:-4]}.pkl")]
    paths.sort(key=lambda p: -os.path.getsize(p))
    Parallel(n_jobs=2)(delayed(R.run_one)(p, out_dir) for p in paths)
    print("done")
