"""
Recomputes the standardised linear coefficients reported in section 2.4 of the
physics note.

Written because none of the eleven coefficients in that table appeared anywhere
in VERIFIED_NUMBERS.md, and the table is the sole support for the claim that
side-chain volume outranks hydrophobicity.

Protocol, stated so the numbers can be reproduced exactly:
  structures  the 62 Amyloid Explorer *-residue.pdb files with usable FoldX
              annotations (2NAO, 2E8D, 8C50 and 8ENQ excluded for constant
              B-factor columns)
  features    the 13 base features of amyfold v0.2, WITHOUT the pairwise
              expansion, since coefficients are only interpretable in the
              linear model
  scaling     features and target both z-scored, so coefficients are directly
              comparable to one another
  fit         ridge, lambda = 50, on all 62 structures pooled (4,710 residues)

Sign convention: FoldX scores stability negatively, so a NEGATIVE coefficient
means the feature is stabilising.
"""
import os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'amyfold'))
from amyfold import parse_pdb, compute_features   # noqa: E402

# Structure search path. Set AMYFOLD_STRUCTURES to a directory of
# Amyloid Explorer *-residue.pdb files, or drop them next to this script.
_HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = [d for d in [
    _HERE,
    os.path.join(_HERE, 'structures'),
    os.environ.get('AMYFOLD_STRUCTURES'),
] if d]


EXCLUDE = {'2NAO', '2E8D', '8C50', '8ENQ'}
LAMBDA = 50.0
SEARCH = SEARCH_DIRS

NAMES = ['hydrophobic contact', 'electrostatics', 'aromatic stacking',
         'H-bond ladder', 'weighted burial', 'side-chain entropy',
         'buried polar', 'exposed apolar', 'side-chain volume', 'proline',
         'glycine', 'hydropathy', 'backbone H-bonds']


def foldx(path):
    acc = {}
    for line in open(path):
        if line.startswith('ENDMDL'):
            break
        if not line.startswith('ATOM') or line[16] not in (' ', 'A'):
            continue
        acc.setdefault((line[21], int(line[22:26])), []).append(float(line[60:66]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


seen = {}
for base in SEARCH:
    for p in glob.glob(os.path.join(base, '*-residue.pdb')):
        seen.setdefault(os.path.basename(p).replace('-residue.pdb', ''), p)

X, Y, used = [], [], []
for pdb, path in sorted(seen.items()):
    if pdb in EXCLUDE:
        continue
    try:
        res = parse_pdb(path)
        pos, aa, xb = compute_features(res)
        fx = foldx(path)
    except Exception:
        continue
    keys = list(res.keys())
    y = (np.array([fx.get(keys[i], np.nan) for i in range(len(pos))])
         if len(keys) == len(pos) else np.full(len(pos), np.nan))
    if np.isnan(y).any():
        first = sorted({k[0] for k in fx})[0]
        y = np.array([fx.get((first, q), np.nan) for q in pos])
    ok = ~np.isnan(y)
    if ok.sum() < 15 or np.nanstd(y) < 1e-9:
        continue
    X.append(xb[ok])
    Y.append(y[ok])
    used.append(pdb)

X = np.vstack(X)
Y = np.concatenate(Y)
names = NAMES[:X.shape[1]]

mu, sd = X.mean(0), X.std(0)
sd[sd == 0] = 1.0
Z = (X - mu) / sd
yz = (Y - Y.mean()) / Y.std()
w = np.linalg.solve(Z.T @ Z + LAMBDA * np.eye(Z.shape[1]), Z.T @ yz)

print(f'{len(used)} structures, {len(Y)} residues, {X.shape[1]} features, '
      f'lambda = {LAMBDA:g}')
print('\nstandardised coefficients, negative = stabilising\n')
for n, c in sorted(zip(names, w), key=lambda t: t[1]):
    bar = '#' * int(abs(c) * 60)
    print(f'   {n:22s} {c:+.3f}  {bar}')

order = [n for n, _ in sorted(zip(names, w), key=lambda t: t[1])]
print(f'\nside-chain volume rank {order.index("side-chain volume") + 1} of {len(order)}')
print(f'hydropathy rank {order.index("hydropathy") + 1}, '
      f'hydrophobic contact rank {order.index("hydrophobic contact") + 1}')
print(f'largest destabilising: {order[-1]}')
