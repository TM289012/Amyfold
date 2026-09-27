"""
Recompute the unfitted three-term model and the two baselines on the SAME
62-structure, 18-family set used for the fitted rows, so every row of the
table in section 2.2/2.3 is comparable.

Previously these three rows were scored on 51 structures and 15 families and
carried a caveat saying so. This removes the caveat.

Nothing is fitted here. The three-term model uses the weights set on physical
grounds; the baselines use no weights at all.
"""
import os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "amyfold"))
from amyfold import parse_pdb, compute_features, FEATURES      # noqa

# Structure search path. Set AMYFOLD_STRUCTURES to a directory of
# Amyloid Explorer *-residue.pdb files, or drop them next to this script.
_HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = [d for d in [
    _HERE,
    os.path.join(_HERE, 'structures'),
    os.environ.get('AMYFOLD_STRUCTURES'),
] if d]


SEARCH = SEARCH_DIRS
EXCLUDE = {"2NAO", "2E8D", "8C50", "8ENQ"}       # constant B-factor columns


def structures():
    seen = {}
    for b in SEARCH:
        for p in glob.glob(os.path.join(b, "*-residue.pdb")):
            k = os.path.basename(p).replace("-residue.pdb", "")
            seen.setdefault(k, p)
    return {k: v for k, v in sorted(seen.items()) if k not in EXCLUDE}


def foldx(path):
    acc = {}
    for line in open(path):
        if line.startswith("ENDMDL"): break
        if not line.startswith("ATOM") or line[16] not in (" ", "A"): continue
        acc.setdefault((line[21], int(line[22:26])), []).append(float(line[60:66]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    def rank(v):
        o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v))
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1: r[m] = r[m].mean()
        return r
    ra, rb = rank(a) - rank(a).mean(), rank(b) - rank(b).mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d else 0.0


KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
      'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
      'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}

iH, iA, iL, iB = (FEATURES.index(x) for x in ("hyd", "aro", "lad", "bur"))

rows = {"three-term (unfitted)": [], "hydropathy lookup": [], "burial alone": []}
used = 0
for name, path in structures().items():
    try:
        pos, aas, X = compute_features(parse_pdb(path))
    except Exception:
        continue
    fx = foldx(path)
    bynum = {}
    for (c, r), v in fx.items():
        bynum.setdefault(r, []).append(v)
    y, k3, kkd, kbu = [], [], [], []
    for i, r in enumerate(pos):
        if r not in bynum: continue
        y.append(float(np.mean(bynum[r])))
        k3.append(X[i, iH] + X[i, iA] + X[i, iL])   # three chemistry terms, equal weight
        kkd.append(KD.get(aas[i], 0.0))
        kbu.append(X[i, iB])
    if len(y) < 8: continue
    used += 1
    rows["three-term (unfitted)"].append(abs(spearman(k3, y)))
    rows["hydropathy lookup"].append(abs(spearman(kkd, y)))
    rows["burial alone"].append(abs(spearman(kbu, y)))

rng = np.random.default_rng(7)
print(f"scored on {used} structures\n")
print(f"{'row':26s} {'mean |rho|':>10s}  {'95% CI':>18s}   previously")
prev = {"three-term (unfitted)": "0.636 (51 structures)",
        "hydropathy lookup": "0.402 (51 structures)",
        "burial alone": "0.130 (51 structures)"}
for lab, v in rows.items():
    v = np.array(v)
    bs = np.array([rng.choice(v, len(v), True).mean() for _ in range(20000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{lab:26s} {v.mean():10.3f}  [{lo:.3f}, {hi:.3f}]   {prev[lab]}")
