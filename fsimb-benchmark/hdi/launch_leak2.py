import sys, pickle, glob, os
from joblib import Parallel, delayed
import run_leak2 as L
from run_wrapper import CEILING

if __name__ == "__main__":
    ALL = [p for p in sorted(glob.glob("data/v2/*.npz")) if os.path.basename(p)[:-4] not in CEILING]
    NULL = ["gordon", "tian", "shipp", "gravier", "golub", "alon", "pomeroy", "GLI_85", "SMK_CAN_187_ir5", "arcene_ir5", "BASEHOCK_ir5", "lung"]
    NPERM = 15
    jobs = [(p, -1) for p in ALL] + [(f"data/v2/{n}.npz", k) for n in NULL for k in range(NPERM)]
    jobs.sort(key=lambda j: -os.path.getsize(j[0]))
    out = Parallel(n_jobs=2, verbose=5)(delayed(L.job)(p, k) for p, k in jobs)
    pickle.dump(out, open("results_leak2.pkl", "wb"))
    print("done")
