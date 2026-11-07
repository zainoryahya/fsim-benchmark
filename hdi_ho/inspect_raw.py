import sys, numpy as np, rdata, scipy.io as sio, warnings
warnings.filterwarnings("ignore")
raw = sys.argv[1]
for name in ["burczynski","chin","christensen","pomeroy","sorlie","subramanian","yeoh","chiaretti"]:
    o = rdata.read_rda(f"{raw}/{name}.RData")[name]
    X = np.asarray(o["x"], float); y = np.asarray(o["y"]).astype(str)
    l, c = np.unique(y, return_counts=True)
    print("DM", name, X.shape, dict(zip(l, c)), "nan", int(np.isnan(X).sum()))
for name in ["SMK_CAN_187","GLI_85","Prostate_GE","CLL_SUB_111","TOX_171","arcene","lung","BASEHOCK","PCMAC","RELATHE","lymphoma","nci9","Carcinom","gisette"]:
    m = sio.loadmat(f"{raw}/skf/{name}.mat")
    X = m["X"]; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    y = np.asarray(m["Y"]).ravel()
    l, c = np.unique(y, return_counts=True)
    print("SK", name, X.shape, dict(zip(l.tolist(), c.tolist())))
