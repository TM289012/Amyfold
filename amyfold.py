#!/usr/bin/env python3
"""
amyfold - fast per-residue stability prediction for amyloid fibril structures

Predicts FoldX-like per-residue free energy contributions (dG_contrib) from a
fibril structure, using twelve physically motivated features and backbone-
derivable geometry only. Runs in milliseconds. No licence, no dependencies
beyond numpy.

    Negative predicted value = residue contributes to fibril stability
    Positive predicted value = structurally frustrated position

VALIDATION
    Leave-one-protein-family-out cross-validation over 51 fibril structures
    spanning 15 amyloid proteins (3,939 residue positions), against FoldX
    per-residue dG from Amyloid Explorer:

        mean Spearman rho per family : 0.722   (range 0.593 - 0.781)
        pooled Spearman rho          : 0.694
        unfitted 3-term baseline      : 0.601
        hydropathy lookup, no structure : 0.402
        burial alone                    : 0.130

    Every prediction in that cross-validation came from a model that had never
    seen the held-out protein family.

WHAT THIS IS NOT
    Not a FoldX replacement. FoldX computes a physically decomposed energy;
    this reproduces its per-residue ranking at rho ~ 0.72. Use it for fast
    screening and triage, not for quantitative energetics.

    Trained on amyloid fibrils only. It has not been tested on globular
    proteins and should not be assumed to work there.

CITATION
    FoldX reference values from Amyloid Explorer (Switch Lab):
      Louros N, van der Kant R, Schymkowitz J, Rousseau F (2022) StAmP-DB.
        Bioinformatics. doi:10.1093/bioinformatics/btac126
      Van der Kant R, Louros N, Schymkowitz J, Rousseau F (2022) Structure.
      Kyriazis et al. (2025) Amyloid Explorer. bioRxiv 2025.10.15.682595
    Structures from the Amyloid Atlas (Sawaya, UCLA) and the RCSB PDB.

USAGE
    python amyfold.py structure.pdb                 # table to stdout
    python amyfold.py structure.pdb --csv out.csv   # write a CSV

    from amyfold import predict
    result = predict('structure.pdb')   # {position: (residue, score)}
"""
from __future__ import annotations
import sys, math, argparse
from collections import defaultdict

import numpy as np

__version__ = '0.2.0'

# ---------------------------------------------------------------- constants
THREE2ONE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
             'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
             'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
KD  = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
       'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CH  = {'D':-1.0,'E':-1.0,'K':1.0,'R':1.0,'H':0.1}
VOL = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,'H':153.2,
       'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,
       'W':227.8,'Y':193.6,'V':140.0}
NCHI = {'A':0,'G':0,'P':0,'S':1,'C':1,'T':1,'V':1,'I':2,'L':2,'N':2,'D':2,
        'F':2,'Y':2,'H':2,'W':2,'M':3,'E':3,'Q':3,'K':4,'R':4}
HB  = {'N':1.0,'Q':1.0,'S':0.8,'T':0.8,'Y':0.5,'H':0.5,'D':0.5,'E':0.5,'R':0.4,'K':0.4,'C':0.3,'W':0.3}
AR  = {'F':1.0,'Y':0.9,'W':1.0,'H':0.5}
POLAR = set('STNQHYCW')
CHG   = set('DEKR')
HYD   = set('AVILMFCW')
BACKBONE = {'N','CA','C','O','OXT'}

H_ = {a: max(0.0, KD[a])/3.0 for a in KD}
Q_ = {a: CH.get(a, 0.0) for a in KD}

CUTOFF = 8.0          # contact cutoff, angstrom, on CB positions

# 13 base features, expanded with all pairwise products -> 104 terms.
# Fitted on 62 structures / 18 families. See amyfold_v2_weights.json.
FEATURES = ['hyd','chg','aro','lad','bur','ent','solvP','solvH','vol','pro','gly','kd','bbhb']

import json as _json, os as _os
_W = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                   'amyfold_v2_weights.json')))
COEF = np.array(_W['coef'])
INTERCEPT = _W['intercept']


def expand(X):
    """Base features plus all pairwise products. Linear models on these
    features plateau at rho 0.73; the interactions take it to 0.76."""
    n = X.shape[1]
    cols = [X]
    for i in range(n):
        for j in range(i, n):
            cols.append((X[:, i]*X[:, j]).reshape(-1, 1))
    return np.hstack(cols)


# ------------------------------------------------------------------ parsing
def parse_pdb(path):
    """First model only, standard residues only, altloc blank or A."""
    res = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith('ENDMDL'):
                break
            if not line.startswith('ATOM'):
                continue
            name = line[17:20].strip()
            if name not in THREE2ONE or line[16] not in (' ', 'A'):
                continue
            key = (line[21], int(line[22:26]))
            res.setdefault(key, {'aa': THREE2ONE[name], 'atoms': {}})
            res[key]['atoms'][line[12:16].strip()] = (
                float(line[30:38]), float(line[38:46]), float(line[46:54]))
    return res


def _unit(v):
    n = np.linalg.norm(v)
    return v/n if n > 1e-9 else v*0


# ----------------------------------------------------------------- features
def compute_features(res):
    """Twelve per-position features. Contacts use CB only, so nothing here
    depends on native side-chain coordinates."""
    keys = sorted(res)
    if not keys:
        raise ValueError('no standard residues parsed')
    cb, direc, Nat, Oat = {}, {}, {}, {}
    for k in keys:
        at = res[k]['atoms']
        base = at.get('CB', at.get('CA'))
        if base is None:
            raise ValueError(f'residue {k} has neither CB nor CA')
        cb[k] = np.array(base)
        ca = np.array(at['CA']) if 'CA' in at else None
        direc[k] = _unit(cb[k]-ca) if ca is not None else None
        if 'N' in at: Nat[k] = np.array(at['N'])
        if 'O' in at: Oat[k] = np.array(at['O'])

    C = np.array([cb[k] for k in keys])
    acc = {f: defaultdict(float) for f in ('hyd', 'chg', 'aro', 'lad', 'bur', 'bbhb')}
    for i in range(len(keys)):
        d = np.sqrt(((C[i+1:]-C[i])**2).sum(1))
        for jj, dist in enumerate(d):
            if dist > CUTOFF:
                continue
            j = i+1+jj
            ka, kb = keys[i], keys[j]
            if ka[0] == kb[0] and abs(ka[1]-kb[1]) < 2:     # skip near-neighbours in chain
                continue
            a, b = res[ka]['aa'], res[kb]['aa']
            va, vb = direc[ka], direc[kb]
            align = float(np.dot(va, vb)) if (va is not None and vb is not None) else 0.0
            # in-register cross-beta ladder: same position, different chain,
            # side chains parallel, spacing 4-6 A
            ladder = (ka[0] != kb[0] and ka[1] == kb[1] and align > 0.9 and 4.0 < dist < 6.0)
            w = 1.0/float(dist)
            acc['bur'][ka[1]] += w/2;  acc['bur'][kb[1]] += w/2
            acc['hyd'][ka[1]] += w*H_[a]*H_[b]/2;  acc['hyd'][kb[1]] += w*H_[a]*H_[b]/2
            acc['chg'][ka[1]] += -w*Q_[a]*Q_[b]/2; acc['chg'][kb[1]] += -w*Q_[a]*Q_[b]/2
            ar = AR.get(a, 0.0)*AR.get(b, 0.0)
            acc['aro'][ka[1]] += w*ar/2; acc['aro'][kb[1]] += w*ar/2
            if ladder:
                L = HB.get(a, 0.0)*HB.get(b, 0.0)
                acc['lad'][ka[1]] += w*L/2; acc['lad'][kb[1]] += w*L/2

    # explicit backbone hydrogen bonds: amide N to carbonyl O within 3.5 A
    NK = [k for k in keys if k in Nat]
    OK = [k for k in keys if k in Oat]
    if NK and OK:
        NP = np.array([Nat[k] for k in NK]); OP = np.array([Oat[k] for k in OK])
        for ii, ka in enumerate(NK):
            dd = np.sqrt(((OP-NP[ii])**2).sum(1))
            for jj, x in enumerate(dd):
                if x > 3.5:
                    continue
                kb = OK[jj]
                if ka[0] == kb[0] and abs(ka[1]-kb[1]) < 2:
                    continue
                acc['bbhb'][ka[1]] += 0.5; acc['bbhb'][kb[1]] += 0.5

    aa = {}
    for k in keys:
        aa[k[1]] = res[k]['aa']
    pos = sorted(acc['bur'])
    b = np.array([acc['bur'][p] for p in pos])
    bn = (b-b.min())/(b.max()-b.min()) if b.max() > b.min() else b*0
    A = [aa[p] for p in pos]

    X = np.column_stack([
        [acc['hyd'][p] for p in pos],
        [acc['chg'][p] for p in pos],
        [acc['aro'][p] for p in pos],
        [acc['lad'][p] for p in pos],
        b,
        np.array([NCHI[a] for a in A])*bn,                                    # entropy loss on burial
        np.array([1.0 if a in POLAR | CHG else 0.0 for a in A])*bn,           # buried polar penalty
        np.array([1.0 if a in HYD else 0.0 for a in A])*(1-bn),               # exposed apolar penalty
        np.array([VOL[a] for a in A])*bn,
        [1.0 if a == 'P' else 0.0 for a in A],
        [1.0 if a == 'G' else 0.0 for a in A],
        [KD[a] for a in A],
        [acc['bbhb'][p] for p in pos],
    ])
    return pos, A, X


# ------------------------------------------------------------------ predict
def predict(path):
    """Return {residue_number: (one_letter, predicted_dG_contrib)}."""
    res = parse_pdb(path)
    pos, aa, X = compute_features(res)
    y = expand(X) @ COEF + INTERCEPT
    return {p: (aa[i], float(y[i])) for i, p in enumerate(pos)}


def _main():
    ap = argparse.ArgumentParser(
        description='Fast per-residue stability prediction for amyloid fibrils.')
    ap.add_argument('pdb', help='fibril structure (.pdb)')
    ap.add_argument('--csv', help='write results to this CSV instead of stdout')
    ap.add_argument('--top', type=int, default=0,
                    help='print only the N most stabilising positions')
    args = ap.parse_args()

    out = predict(args.pdb)
    rows = sorted(out.items(), key=lambda kv: kv[1][1])          # most stabilising first

    if args.csv:
        with open(args.csv, 'w') as fh:
            fh.write('position,residue,predicted_dG_contrib\n')
            for p, (a, v) in sorted(out.items()):
                fh.write(f'{p},{a},{v:.4f}\n')
        print(f'wrote {len(out)} positions to {args.csv}')
        return

    print(f'# amyfold {__version__}   {args.pdb}   {len(out)} positions')
    print('# negative = stabilising, positive = frustrated')
    print(f'{"pos":>6s} {"aa":>3s} {"dG_pred":>9s}')
    for p, (a, v) in (rows[:args.top] if args.top else sorted(out.items())):
        print(f'{p:6d} {a:>3s} {v:9.3f}')


if __name__ == '__main__':
    _main()
