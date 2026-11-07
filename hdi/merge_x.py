"""Merge the original 2-repeat results (r = 0, 1) with the extra 3-repeat results (r = 2..4) into 5-repeat results.

Metrics are per-repeat means, so the merge is a weighted mean (2/5, 3/5); subset lists and per-run info lists are
concatenated in repeat order. Keys present only in the original (configurations that were not re-run) are kept
unchanged and listed in the pickle under 'unmerged'. Originals are moved to <dir>_r2 / <file>_r2.pkl first.
Usage: python merge_x.py <suite dir>
"""
import glob, os, pickle, shutil, sys
import numpy as np

WA, WB = 2 / 5, 3 / 5
UNMERGED = set()


def merge(a, b, path=""):
    if isinstance(a, dict):
        out = {}
        for k in a:
            if k in b:
                out[k] = merge(a[k], b[k], f"{path}/{k}")
            else:
                out[k] = a[k]; UNMERGED.add(f"{path}/{k}")
        return out
    if isinstance(a, list):
        return a + b
    if isinstance(a, (bool, str, np.ndarray)) or a is None:
        return a
    if isinstance(a, (int, float, np.floating, np.integer)):
        return a if a == b and isinstance(a, (int, np.integer)) else float(WA * a + WB * b)
    return a


if __name__ == "__main__":
  root = sys.argv[1]
  os.chdir(root)
  for d in ["results_wrap", "results_wrapH", "results_cap", "results_emt", "results_xclf", "results_ridge"]:
      if not os.path.isdir(d + "_x"):
          print("skip", d); continue
      if not os.path.isdir(d + "_r2"):
          shutil.move(d, d + "_r2"); os.makedirs(d)
      n = 0
      for f in sorted(glob.glob(f"{d}_r2/*.pkl")):
          name = os.path.basename(f)
          fx = f"{d}_x/{name}"
          if not os.path.exists(fx):
              shutil.copy(f, f"{d}/{name}"); continue
          UNMERGED = set()
          a, b = pickle.load(open(f, "rb")), pickle.load(open(fx, "rb"))
          m = merge(a, b)
          m["unmerged"] = sorted(UNMERGED); m["nrep"] = 5
          pickle.dump(m, open(f"{d}/{name}", "wb")); n += 1
      print(d, "merged", n)
  for f in ["results_thr.pkl", "results_tomek.pkl"]:
      fx = f.replace(".pkl", "_x.pkl")
      if not os.path.exists(fx):
          print("skip", f); continue
      r2 = f.replace(".pkl", "_r2.pkl")
      if not os.path.exists(r2):
          shutil.move(f, r2)
      UNMERGED = set()
      a, b = pickle.load(open(r2, "rb")), pickle.load(open(fx, "rb"))
      pickle.dump(merge(a, b), open(f, "wb"))
      print(f, "merged; unmerged keys:", len(UNMERGED))
