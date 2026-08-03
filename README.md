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
18 amyloid proteins**, against FoldX per-residue ΔG from Amyloid Explorer. To
predict any family, the model was fit on the other 17 and that family was hidden.

| | Spearman ρ vs FoldX |
|---|---|
| **amyfold 0.2** | **0.760**, 95% CI [0.739, 0.780] |
| amyfold 0.1 (linear, 12 features) | 0.722 |
| Unfitted 3-term physical model | 0.601 |
| Residue hydropathy lookup, no structure | 0.402 |
| Burial alone | 0.130 |

Positive on **62 of 62** structures. Range 0.516 to 0.923.

### Generalisation to unseen protein families

Before v0.2 was fit, v0.1 was tested on 11 structures from 10 protein families
absent from its training set entirely: lysozyme, RIPK, β-lactoglobulin, LECT2,
TCERG1, Sup35 (yeast prion), PI3K, αA-crystallin, apolipoprotein A-II and a
plant amylase inhibitor.

| | ρ |
|---|---|
| Held-out families, per-structure mean | **0.724** |
| Cross-validated on trained families | 0.722 |

No degradation outside the training distribution. Those 11 structures are
included in v0.2's training set, so the 0.760 above is the honest current
estimate and comes from the leave-one-family-out protocol.

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
chain. No native side-chain coordinates are used, so the features contain no
information a backbone-only method would lack.

Thirteen base features per position: hydrophobic contact, electrostatic contact,
aromatic stacking, in-register hydrogen-bond ladder, weighted burial, side-chain
entropy on burial, buried-polar penalty, exposed-apolar penalty, volume on
burial, proline, glycine, hydropathy, and explicit backbone hydrogen bonds
(amide N to carbonyl O within 3.5 Å).

Those are expanded with all pairwise products to 104 terms and fitted by ridge
regression, λ = 50. The interactions matter: the same features fitted linearly
reach 0.733, the expansion takes it to 0.760.

### What the fit says physically

From the linear version, where coefficients are interpretable. Ranked by
standardised weight:

| Stabilising | | Destabilising | |
|---|---|---|---|
| side-chain volume | −0.441 | **buried polar** | **+0.350** |
| proline | −0.330 | burial alone | +0.172 |
| hydropathy | −0.309 | glycine | +0.128 |
| hydrophobic contact | −0.283 | | |
| aromatic stacking | −0.251 | | |
| H-bond ladder | −0.245 | | |
| side-chain entropy | −0.169 | | |
| electrostatics | −0.149 | | |

Two things worth noting. **Side-chain volume outranks hydrophobicity** as a
predictor of fibril stability. And the side-chain entropy term, though it takes a
real weight, contributes **exactly nothing** in ablation, because chi-angle count
and side-chain volume are nearly the same variable.

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
