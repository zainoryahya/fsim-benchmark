import pickle, glob, os
from joblib import Parallel, delayed
import run_thr as T
from run_wrapper import CEILING
if __name__ == "__main__":
    paths = [p for p in sorted(glob.glob("data/v2/*.npz")) if os.path.basename(p)[:-4] not in CEILING]
    paths.sort(key=lambda p: -os.path.getsize(p))
    out = Parallel(n_jobs=2, verbose=5)(delayed(T.job)(p) for p in paths)
    pickle.dump(dict(out), open("results_thr.pkl", "wb"))
    print("done")
