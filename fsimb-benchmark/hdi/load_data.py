import sys, numpy as np, rdata
src, dst = sys.argv[1], sys.argv[2]
for name in ["alon","gordon","shipp","golub","chowdary","gravier","singh","tian","west"]:
    conv = rdata.read_rda(f"{src}/{name}.RData")
    obj = conv[name]
    x = obj["x"]; y = obj["y"]
    X = np.asarray(x, dtype=float)
    yv = np.asarray(y.astype(str) if hasattr(y,'astype') else y)
    labs, cnt = np.unique(yv, return_counts=True)
    minority = labs[np.argmin(cnt)]
    yb = (yv == minority).astype(int)
    np.savez_compressed(f"{dst}/{name}.npz", X=X, y=yb)
    print(f"{name}: n={X.shape[0]} p={X.shape[1]} classes={dict(zip(labs,cnt))} minority={minority} IR={cnt.max()/cnt.min():.2f} nan={np.isnan(X).sum()}")
