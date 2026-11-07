import pickle, glob, os, sys
from joblib import Parallel, delayed
import run_tomek as T
CEILING = {"chiaretti", "CLL_SUB_111", "lymphoma", "Carcinom"}
if __name__ == "__main__":
    data_dir, out, nj = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2
    paths = [p for p in sorted(glob.glob(f"{data_dir}/*.npz"), key=lambda p: -os.path.getsize(p))
             if os.path.basename(p)[:-4] not in CEILING]
    res = Parallel(n_jobs=nj, verbose=5)(delayed(T.job)(p) for p in paths)
    pickle.dump(dict(res), open(out, "wb"))
    print("done")
