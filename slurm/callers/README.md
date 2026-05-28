# slurm_kinsim/callers/ — methylation caller scripts

Scripts génériques qui consomment un BAM aligné + ref et produisent un
`motifs.csv` (format combined-parser : `mod_type,motif,offset,frac_mod,n_sites,source`).

Appelables depuis n'importe quel dataset via arguments positionnels.

| Script | Mod types | Input | Output |
|---|---|---|---|
| `ipdsummary.slurm` | m6A, m4C | aligned BAM (bystrandified) + ref | `*_ipdSummary.gff` + `*_ipdSummary.csv` |
| `pbmotifmaker.slurm` | — (discovery) | ipdSummary GFF + ref | `motifs_ipdsummary.csv` (PacBio format) |
| `jasmine_modkit.slurm` | m5C (CpG) | raw BAM with C+m MM tags + ref | `motifs_jasmine.csv` (combined format) |
| `merge_motifs.slurm` | all | 2-3 motifs.csv | `motifs_merged.csv` (combined format) |

## Merge policy — just a threshold

Pas de règles de précédence alambiquées. La logique est simple :

1. Chaque caller produit un `motifs.csv` au format combined-parser
2. `merge_motifs.slurm` concatène tous les fichiers d'entrée
3. Filtre : **`frac_mod >= 0.7`** (threshold configurable)
4. Déduplique par `(motif, offset)` — si le même motif apparaît plusieurs fois, garde la plus haute fraction

**Pourquoi ça marche sans règle** : les mislabels ipdSummary du 5mC en m4C ressortent avec des fractions faibles (~25-50 %, signal 5mC atténué). Le threshold 0.7 les filtre naturellement. Les vrais calls (m6A d'ipdSummary à 99 %, m5C de jasmine à 99 %, vrai m4C à 94 %) passent.

```
bc2071 merge example (threshold=0.7):
  ipdSummary m4C  TCGCGA       94%   → PASS   (vrai m4C)
  ipdSummary m4C  CCCGCCCC    100%   → PASS   (vrai m4C ou m5C, on s'en fout du label)
  ipdSummary m4C  GCCGGCYR     34%   → FAIL   (mislabel 5mC, dégage)
  jasmine   m5C   GCCNGC       99%   → PASS
```

## Why not put KinMethyl here yet ?

Pending validation on our P1-C1 data (bc2046 test). Will add `kinmethyl.slurm`
once the chemistry mismatch question is answered.
