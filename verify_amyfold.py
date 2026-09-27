"""
Independent recomputation of the amyfold validation numbers.

Written because two claims in the physics note were never entered into
VERIFIED_NUMBERS.md and so had never been checked:

  (a) mean per-structure rho of 0.760 under leave-one-protein-family-out
  (b) 0.724 on "eleven structures from ten protein families absent from the
      training set altogether"

Claim (b) is attributed in the amyfold docstring to model v0.1, not to the
released v0.2, and no saved output for it exists. This script recomputes both
for the CURRENT model and adds a strict version of (b): train only on the
seventeen named amyloid families, test on the singleton proteins that have no
family-mates in the set at all.

Reads FoldX per-residue dG from the B-factor column of the Amyloid Explorer
*-residue.pdb files, exactly as the model does.
"""
import os, sys, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'amyfold'))

from amyfold import parse_pdb, compute_features, expand   # noqa: E402

# Structure search path. Set AMYFOLD_STRUCTURES to a directory of
# Amyloid Explorer *-residue.pdb files, or drop them next to this script.
_HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = [d for d in [
    _HERE,
    os.path.join(_HERE, 'structures'),
    os.environ.get('AMYFOLD_STRUCTURES'),
] if d]

try:
    from fam import family                                 # noqa: E402
except ImportError:
    def family(_):
        return 'other'

SEARCH = SEARCH_DIRS

# constant B-factor columns: FoldX values were never computed for these
EXCLUDE = {'2NAO', '2E8D', '8C50'}

LAMBDA = 50.0
rng = np.random.default_rng(37)


def find_structures():
    seen = {}
    for base in SEARCH:
        for p in glob.glob(os.path.join(base, '*-residue.pdb')):
            k = os.path.basename(p).replace('-residue.pdb', '')
            seen.setdefault(k, p)
    return {k: v for k, v in sorted(seen.items()) if k not in EXCLUDE}


def foldx_from_bfactor(path):
    """Mean B-factor per (chain, resnum), which is where Amyloid Explorer
    stores the FoldX per-residue dG contribution."""
    acc = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith('ENDMDL'):
                break
            if not line.startswith('ATOM') or line[16] not in (' ', 'A'):
                continue
            key = (line[21], int(line[22:26]))
            acc.setdefault(key, []).append(float(line[60:66]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def spearman(a, b):
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / den) if den else 0.0


def ridge(X, y, lam):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam
    return np.linalg.solve(A, Xb.T @ y)


def apply(w, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


# ------------------------------------------------------------------ build
atlas = {}
for base in SEARCH:
    p = os.path.join(base, 'atlas.json')
    if os.path.exists(p):
        atlas = json.load(open(p))
        break

data = []
skipped = []
for pdb, path in find_structures().items():
    try:
        res = parse_pdb(path)
        pos, aa, Xb = compute_features(res)
        fx = foldx_from_bfactor(path)
    except Exception as e:
        skipped.append((pdb, f'{type(e).__name__}'))
        continue
    keys = list(res.keys())
    y = np.array([fx.get(keys[i], np.nan) for i in range(len(pos))]) \
        if len(keys) == len(pos) else None
    if y is None or np.isnan(y).any():
        # fall back: match on residue number within the first chain
        first = sorted({k[0] for k in fx})[0]
        y = np.array([fx.get((first, p), np.nan) for p in pos])
    ok = ~np.isnan(y)
    if ok.sum() < 15 or np.nanstd(y) < 1e-9:
        skipped.append((pdb, 'constant or too few FoldX values'))
        continue
    prot = atlas.get(pdb, {}).get('prot', '')
    f = family(prot)
    if f == 'other':
        f = f'other:{pdb}'          # singleton, no family-mates
    data.append(dict(pdb=pdb, fam=f, prot=prot,
                     X=expand(Xb[ok]), y=y[ok], n=int(ok.sum())))

fams = sorted({d['fam'] for d in data})
named = sorted({f for f in fams if not f.startswith('other:')})
singles = sorted({f for f in fams if f.startswith('other:')})
print(f'structures used {len(data)}   skipped {len(skipped)}')
for p, why in skipped:
    print(f'   skipped {p}: {why}')
print(f'named families {len(named)}: {", ".join(named)}')
print(f'singleton proteins {len(singles)}')
print()


def loro(train_pred):
    """train_pred(held_out_family) -> list of (pdb, rho)"""
    out = []
    for d in data:
        w = train_pred(d['fam'])
        out.append((d['pdb'], d['fam'], spearman(apply(w, d['X']), d['y'])))
    return out


def fit_excluding(fam):
    tr = [d for d in data if d['fam'] != fam]
    X = np.vstack([d['X'] for d in tr])
    y = np.concatenate([d['y'] for d in tr])
    return ridge(X, y, LAMBDA)


print('=' * 68)
print('A. LEAVE-ONE-FAMILY-OUT, all families including singletons')
print('=' * 68)
rows = loro(fit_excluding)
r = np.array([abs(x[2]) for x in rows])
bs = np.array([rng.choice(r, len(r), True).mean() for _ in range(10000)])
print(f'   structures {len(r)}   families {len(fams)}')
print(f'   mean |rho| {r.mean():.3f}   95% CI [{np.percentile(bs,2.5):.3f}, '
      f'{np.percentile(bs,97.5):.3f}]')
print(f'   median {np.median(r):.3f}   range {r.min():.3f} to {r.max():.3f}')
pos = [x for x in rows if x[2] > 0]
print(f'   positive correlation on {len(pos)} of {len(rows)}')

print()
print('=' * 68)
print('B. STRICT HOLDOUT: train on named families only, test on singletons')
print('=' * 68)
tr = [d for d in data if not d['fam'].startswith('other:')]
te = [d for d in data if d['fam'].startswith('other:')]
if te:
    X = np.vstack([d['X'] for d in tr])
    y = np.concatenate([d['y'] for d in tr])
    w = ridge(X, y, LAMBDA)
    rr = [(d['pdb'], d['prot'][:40], abs(spearman(apply(w, d['X']), d['y'])))
          for d in te]
    v = np.array([x[2] for x in rr])
    bs = np.array([rng.choice(v, len(v), True).mean() for _ in range(10000)])
    print(f'   trained on {len(tr)} structures from {len(named)} named families')
    print(f'   tested on {len(te)} structures, each a protein with no '
          f'family-mate in the set')
    print(f'   mean |rho| {v.mean():.3f}   95% CI [{np.percentile(bs,2.5):.3f}, '
          f'{np.percentile(bs,97.5):.3f}]')
    for pdb, prot, rho in sorted(rr, key=lambda z: -z[2]):
        print(f'      {pdb}  {rho:.3f}  {prot}')
    trained = np.array([abs(x[2]) for x in rows
                        if not x[1].startswith('other:')])
    print(f'\n   for comparison, cross-validated on trained families: '
          f'{trained.mean():.3f}')
