"""
How much of the answer does a native-referenced steric term give away?

Section 3 shows that calibrating an excluded-volume term against native
side-chain volumes inflated recovery by roughly forty points in one model. The
question a reader should ask is whether that is peculiar to this model or a
property of the reference state itself.

This measures it at the source, with no model involved. A design score whose
steric reference is written

    room(i,j) = RAD[native(i)] + RAD[native(j)]

has, for every position, access to the native side-chain radius. So the ceiling
on what that reference can leak is simply: how well does native side-chain
volume alone identify the native residue? Nothing here uses a structure, a
contact map, or any energy function. It is an upper bound on the leak available
to ANY score built on a native-volume reference.

Three estimators, in increasing generosity to the scorer:
  exact        volume known exactly, pick the amino acid with that volume
  tolerance t  volume known to within t A^3, pick uniformly among candidates
  volume+comp  as above, broken by amino-acid frequency in the actual structures
"""
import glob, math, os, sys
from collections import Counter
import numpy as np

# Structure search path. Set AMYFOLD_STRUCTURES to a directory of
# Amyloid Explorer *-residue.pdb files, or drop them next to this script.
_HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = [d for d in [
    _HERE,
    os.path.join(_HERE, 'structures'),
    os.environ.get('AMYFOLD_STRUCTURES'),
] if d]


VOL = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,
       'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,
       'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
A3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
      'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
      'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

# native sequences from every fibril structure available.
# Deduplicate by filename: the search path may contain the same structure in
# more than one directory, and an earlier version of this script counted those
# copies twice, inflating the structure and residue totals. The recovery
# percentages are composition-weighted, so duplicates also reweighted them
# slightly. Dedupe first, always.
seqs = []
_done = set()
for base in SEARCH_DIRS:
    for p in sorted(glob.glob(os.path.join(base, "*.pdb"))):
        if os.path.basename(p) in _done:
            continue
        _done.add(os.path.basename(p))
        seen, s = set(), []
        for ln in open(p):
            if ln.startswith("ENDMDL"): break
            if not ln.startswith("ATOM") or ln[12:16].strip() != "CA": continue
            k = (ln[21], int(ln[22:26]))
            if k in seen: continue
            seen.add(k)
            a = A3.get(ln[17:20].strip())
            if a: s.append(a)
        if len(s) >= 10: seqs.append((os.path.basename(p), s))

allres = [a for _, s in seqs for a in s]
comp = Counter(allres)
N = len(allres)
print(f"{len(seqs)} structures, {N} residues\n")

# --- baseline: guess the single most common residue everywhere
mode = comp.most_common(1)[0]
print(f"baseline, always guess the commonest residue ({mode[0]}): "
      f"{100*mode[1]/N:.2f}%")
# --- baseline: guess in proportion to composition (expected accuracy)
prop = sum((c/N)**2 for c in comp.values())
print(f"baseline, guess in proportion to composition:            {100*prop:.2f}%\n")

# --- the leak: identity recoverable from native volume alone
def recover(tol, weight_by_comp):
    hits = 0.0
    for a in allres:
        v = VOL[a]
        cand = [b for b in VOL if abs(VOL[b] - v) <= tol]
        if weight_by_comp:
            w = np.array([comp.get(b, 0) + 1e-9 for b in cand], float)
            p = w / w.sum()
            hits += p[cand.index(a)]
        else:
            hits += 1.0 / len(cand)
    return 100.0 * hits / N

print("identity recoverable from the native side-chain volume alone:")
print(f"{'tolerance (A^3)':>16s}  {'uniform pick':>13s}  {'composition-weighted':>21s}")
for tol in (0.0, 1.0, 2.5, 5.0, 10.0, 20.0):
    print(f"{tol:16.1f}  {recover(tol, False):12.2f}%  {recover(tol, True):20.2f}%")

# --- how much does knowing the SUM of two radii narrow a pair?
RAD = {a: (3*v/(4*math.pi))**(1/3.) for a, v in VOL.items()}
pairs = {}
for a in RAD:
    for b in RAD:
        pairs.setdefault(round(RAD[a]+RAD[b], 3), set()).add(frozenset((a, b)))
amb = [len(v) for v in pairs.values()]
print(f"\nradius sums: {len(pairs)} distinct values over 210 unordered pairs")
print(f"   {sum(1 for x in amb if x == 1)} sums identify the pair uniquely")
print(f"   mean pairs sharing a sum: {np.mean(amb):.2f}, max {max(amb)}")
