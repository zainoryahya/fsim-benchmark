import pickle, glob, os
from joblib import Parallel, delayed
import run_thr as T
if __name__ == "__main__":
    paths = sorted(glob.glob("data/doro/*.npz"), key=lambda p: -os.path.getsize(p))
    out = Parallel(n_jobs=2, verbose=5)(delayed(T.job)(p) for p in paths)
    pickle.dump(dict(out), open("results_doro_thr.pkl", "wb"))
    print("done")
