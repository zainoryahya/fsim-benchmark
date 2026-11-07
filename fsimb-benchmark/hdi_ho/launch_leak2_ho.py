import pickle, glob, os
from joblib import Parallel, delayed
import run_leak2 as L
if __name__ == "__main__":
    ALL = sorted(glob.glob("data/v2/*.npz"))
    jobs = [(p, -1) for p in ALL] + [(p, k) for p in ALL for k in range(10)]
    jobs.sort(key=lambda j: -os.path.getsize(j[0]))
    out = Parallel(n_jobs=2, verbose=5)(delayed(L.job)(p, k) for p, k in jobs)
    pickle.dump(out, open("results_leak2.pkl", "wb"))
    print("done")
