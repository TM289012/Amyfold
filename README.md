# amyfold

Fast per-residue stability prediction for amyloid fibril structures.

Takes a fibril PDB, returns a predicted per-residue free-energy contribution for
every position. Runs in milliseconds, needs only numpy, no licence required.

```
python amyfold.py 6CU7.pdb
python amyfold.py 6CU7.pdb --csv out.csv
python amyfold.py 6CU7.pdb --top 10        # 10 most stabilising positions
```

```python
from amyfold import predict
scores = predict('6CU7.pdb')     # {position: (residue, predicted_dG)}
```

Negative means the residue contributes to fibril stability. Positive means the
position is structurally frustrated.

---

## Validation

Leave-one-protein-family-out cross-validation across **62 fibril structures,
27 amyloid proteins grouped into 18 families**, against FoldX per-residue ΔG from
Amyloid Explorer. To predict any family, the model was fit on the other 17 and
that family was hidden.

Every row below is scored on the same 62 structures, so the rows are directly
comparable to each other.

| | Spearman ρ vs FoldX | 95% CI |
|---|---|---|
| **amyfold 0.2** (13 features + interactions) | **0.760** | [0.739, 0.780] |
| Same 13 features, fitted linearly | 0.733 | [0.713, 0.752] |
| Unfitted three-term physical model | 0.621 | [0.595, 0.647] |
| Residue hydropathy lookup, no structure | 0.452 | [0.411, 0.491] |
| Burial alone | 0.202 | [0.175, 0.231] |

Positive on **62 of 62** structures. Range 0.516 to 0.923.

An independent recomputation that holds out each of the 27 proteins separately,
rather than grouping the ten singletons into one family, gives 0.752
[0.730, 0.774]. A stricter partition returning a slightly lower number is the
expected behaviour.

amyfold **0.1** reported 0.722 with 12 linear features, and the unfitted three-term
model 0.636. Both were measured on the earlier 51-structure, 15-family set and are
not comparable to the table above. The baseline rows were recomputed on the full 62
for this release, which is why hydropathy and burial both rose.

### Generalisation to unseen proteins

Leave-one-family-out still allows a related protein into training. Ten of the 62
structures are proteins with no family-mate anywhere in the set:
β-lactoglobulin, PI3K, a CHCHD-domain protein, insulin B chain, LECT2,
αA-crystallin, CPEB, eRF3, a plant α-amylase/trypsin inhibitor and TCERG1.
Fitting only on the 52 structures from the 17 named amyloid families and testing
on those ten:

| | ρ |
|---|---|
| **Ten unrelated proteins, never seen** | **0.770**, 95% CI [0.721, 0.818] |
| Cross-validated on trained families | 0.748 |

No degradation outside the training distribution. Per structure the ten range
from 0.650 (TCERG1) to 0.907 (β-lactoglobulin).

An earlier release of this README reported 0.724 for a similar test on v0.1.
That figure could not be reproduced from any saved output and has been replaced
by the recomputation above, which uses the released v0.2 weights and ships with
the script that produces it.

---

## What it cannot do

Tested and failed. Listed because a tool that only advertises what works is not
trustworthy.

| Task | Result |
|---|---|
| Rank **whole structures** by stability | ρ = 0.13, n = 62, not significant |
| Rank **polymorphs of the same protein** | mean ρ = −0.05 across 6 families |
| Distinguish native sequence from **scrambled** on the same backbone | native wins 7/12; beats all 30 decoys on 1/12 |
| Predict **mutational effects** across peptides | ρ = 0.25-0.32 against deep mutational scanning |

So this ranks positions **within one given structure**. It cannot compare
structures, recognise a native fold, or score variants. Removing the
per-structure burial normalisation was tried as a fix for between-structure
ranking and made it worse (0.13 → 0.03).

Other honest limitations:

- **Not a FoldX replacement.** It reproduces per-residue ranking, not a
  physically decomposed energy.
- **The reference is FoldX, not experiment.** Whatever FoldX gets wrong about
  amyloid, this inherits exactly.
- **All reference values come from one pipeline** (Amyloid Explorer). Systematic
  error there propagates.
- **Feature selection was informed by all 62 structures.** Weights are
  cross-validated; the choice of which features to include is not.
- **Amyloid fibrils only**, ≥ 3 stacked chains. Untested on globular proteins.

---

## Method

Contacts between **Cβ atoms** within 8 Å, excluding near-neighbours in the same
chain. No native side-chain coordinates enter the geometry, so contacts carry no
information about how side chains are actually placed.

The features do use residue identity, through volume, hydropathy, proline and
glycine. That is legitimate here because the sequence is an **input** to the
stability task, not the thing being predicted. It would not be legitimate in
sequence design, where the sequence is the answer. The accompanying physics note
shows what goes wrong when a native-referenced term leaks into a design
benchmark, and it is a large effect, so the distinction is worth stating plainly.

Thirteen base features per position: hydrophobic contact, electrostatic contact,
aromatic stacking, in-register hydrogen-bond ladder, weighted burial, side-chain
entropy on burial, buried-polar penalty, exposed-apolar penalty, volume on
burial, proline, glycine, hydropathy, and explicit backbone hydrogen bonds
(amide N to carbonyl O within 3.5 Å).

Those are expanded with all pairwise products to 104 terms and fitted by ridge
regression, λ = 50. The interactions matter: the same thirteen features fitted
linearly reach 0.733, the expansion takes it to 0.760. Results are insensitive to
λ between 1 and 400.

### What the fit says physically

Coefficients are interpretable only without the pairwise expansion, so all
thirteen base features were refitted linearly with features and target z-scored,
on all 62 structures pooled (4,710 residues, ridge, λ = 50). FoldX scores
stability negatively, so a negative coefficient marks a stabilising feature.

| Stabilising | | Destabilising | |
|---|---|---|---|
| **side-chain volume** | **−0.325** | **buried polar** | **+0.229** |
| proline | −0.233 | weighted burial | +0.192 |
| hydropathy | −0.208 | glycine | +0.082 |
| hydrophobic contact | −0.193 | exposed apolar | −0.006 |
| aromatic stacking | −0.164 | | |
| H-bond ladder | −0.149 | | |
| electrostatics | −0.121 | | |
| side-chain entropy | −0.103 | | |
| backbone H-bonds | −0.096 | | |

**Side-chain volume outranks hydrophobicity**, appearing first while raw
hydropathy is third and hydrophobic contact fourth. That is consistent with tight
steric packing, rather than hydrophobic burial, being the dominant stabilising
interaction in cross-β cores.

Three warnings about reading this table.

**These are coefficients against FoldX, not against experiment.** A large weight
establishes that the feature is needed to reproduce FoldX, not that it captures a
real physical effect.

**The buried-polar coefficient was tested directly and did not survive.** A
pre-registered test against deep mutational scanning of nucleation in two systems
came out contradictory: amyloid-β gave +0.730 with a CI spanning zero, IAPP gave
−0.915 with a CI excluding zero, so the two disagree and IAPP runs the wrong way.
Burial and substitution polarity also turn out to be collinear by construction, so
the test could not separate them. **Treat +0.229 as a property of this fit, not as
an established property of fibrils.**

**Two weights are noise.** Side-chain entropy takes a real coefficient but
removing it changes cross-validated performance by −0.0001, making the model
fractionally better without it, because chi-angle count and side-chain volume
correlate at r = 0.69. The exposed-apolar penalty is indistinguishable from zero.

The full pre-registered test and its failure are written up in the accompanying
physics note.

---

## Reproducing the numbers

Every figure in this README comes from a script in this repository. None of them
need FoldX or a licence; they need numpy and the structures.

```
python verify_amyfold.py        # the 0.760 / 0.752 / 0.770 validation figures
python verify_coefficients.py   # the standardised coefficient table
python baselines_62.py          # the three baseline rows, all on 62 structures
python leak_bound.py            # the reference-state leak bound
```

The scripts look for Amyloid Explorer `*-residue.pdb` files next to themselves,
in `./structures`, or in whatever directory `AMYFOLD_STRUCTURES` points at:

```
AMYFOLD_STRUCTURES=~/amyloid_explorer python verify_amyfold.py
```

Those structures are the Switch Lab's and are not redistributed here. Get them
from Amyloid Explorer, cited below. `6CU7.pdb` is included so the quickstart
above runs out of the box. `leak_bound_manifest.txt` lists every structure used
for the leak bound, file by file, so that figure can be reproduced exactly.

---

## Data and credit

FoldX reference values and standardised structures from **Amyloid Explorer**
(Switch Lab, VIB/KU Leuven and UT Southwestern):

- Louros N, van der Kant R, Schymkowitz J, Rousseau F (2022) *StAmP-DB: A platform for structures of polymorphic amyloid fibril cores.* Bioinformatics. doi:10.1093/bioinformatics/btac126
- Van der Kant R, Louros N, Schymkowitz J, Rousseau F (2022) *Thermodynamic analysis of amyloid fibril structures reveals a common framework for stability in amyloid polymorphs.* Structure.
- Kyriazis et al. (2025) *Amyloid Explorer: a global atlas of amyloid fibril structures and thermodynamic principles.* bioRxiv 2025.10.15.682595

Structures from the **Amyloid Atlas** (Sawaya, UCLA) and the RCSB PDB.
FoldX: Schymkowitz et al., the FoldX force field.

If you use this, please cite the Switch Lab papers above, since the reference
energetics are theirs.
