# KinSim — Document de référence TFE

Récapitulatif complet du projet KinSim pour mon TFE : architecture, design
decisions, modifications majeures, références scientifiques.

**Statut :** vivant — mettre à jour à chaque session de travail.

---

# Partie I — Architecture générale

## I.1 Problème scientifique

PacBio HiFi mesure pour chaque base deux signaux cinétiques :
- **IPD** (Inter-Pulse Duration) : temps entre 2 incorporations de nucléotides
- **PW** (Pulse Width) : durée d'incorporation d'un nucléotide

La méthylation (m6A, m4C, m5C, ...) perturbe ces signaux : la polymérase
"hésite" en présence d'une base modifiée. Les outils existants (ipdSummary,
jasmine) **détectent** cette signature ; aucun n'est conçu pour la
**simuler** depuis zéro.

**KinSim** simule des kinétiques HiFi synthétiques réalistes conditionnées
par la méthylation → permet de générer des reads simulés (via PBSIM3) avec
des signaux IPD/PW informatifs pour l'entraînement et l'évaluation de
binners métagénomiques méthylation-aware.

**Références biologiques :**
- **Flusberg et al., 2010** — "Direct detection of DNA methylation during
  single-molecule, real-time sequencing", *Nature Methods*.
  https://www.nature.com/articles/nmeth.1459
- **Feng et al., 2013** — kineticsTools / `ipdSummary` algorithm, *PLoS
  Computational Biology*.
- **Beaulaurier et al., 2019** — "Metagenomic binning and association of
  plasmids with bacterial host genomes using DNA methylation", *Nature
  Biotechnology*.
- **Tse et al., 2021** — "Methylation profiling at the nucleotide level
  using PacBio HiFi sequencing".
- **Tourancheau et al., 2021** — "Discovering multiple types of DNA
  methylation from bacteria and microbiome using nanopore sequencing".

## I.2 Pipeline KinSim

```
aligned BAMs ─→ extract ─→ refine ─→ train ─→ generate ─→ verify
   (real)        (raw         (clean)    (model)   (BAM out)   (motifs)
                  shards)
```

### I.2.1 `extract`

**Rôle** : depuis un BAM aligné réel + ses motifs détectés (motifs.csv ou
combined.csv), extraire des "samples" (rows) pour l'entraînement.

**Catégorisation** (introduite en mars 2026, cf. `CLAUDE.md`) :
- **BASELINE (cat=0)** : position loin de toute méthylation
  (distance ≥ K = 11 bases). Capped à N=50 par kmer via reservoir sampling
  pour bornage RAM.
- **SLOWED (cat=1)** : position au `signal_offset` d'une méthylation
  (e.g. m6A@+0, m6A@+5). C'est là qu'on attend une vraie signature
  cinétique. Pas de cap (tous les sites de méth émis).
- **NEAR_METH (cat=2)** : position proche d'une méth (≤K) mais PAS à un
  `signal_offset` → cinétique attendue baseline. Contrôle négatif.

**Pourquoi cette catégorisation :** entraîner uniquement sur les méthylations
centrales (cat=0 m6A à GATC) priverait le modèle des signaux décalés
(`+5` pour m6A par exemple). Les rows NEAR_METH apprennent au modèle que
"meth à proximité" sans signal_offset = baseline, pas un boost.

### I.2.2 `refine`

**Rôle** : filtrer les rows SLOWED qui ressemblent à du bruit (vraiment pas
méthylées). Les rows BASELINE et NEAR_METH ne sont jamais filtrées.

**Méthode `gmm`** (default) :
- Pour chaque (meth_type T) présent dans les SLOWED, ajuster un GMM
  2-composantes sur le pool combiné `baseline + slowed_by_T` (baseline
  sous-échantillonné pour matcher le volume slowed)
- Valider : ≥ 85% du baseline doit se concentrer dans le component à plus
  faible moyenne (sinon le fit est cassé, on ne filtre pas)
- Drop les rows slowed dont la posterior dans le composant baseline > 0.5

**Méthode `p95`** (legacy) :
- Seuil global = p95 des per-kmer baseline means
- Même seuil pour tous les types de méth

**Pourquoi GMM par défaut :** le p95 global est injuste pour les kmers à
baseline élevée (GC-rich) — beaucoup de leurs rows SLOWED tombent en dessous
du seuil global mais SONT réellement méthylés. GMM par-(T) laisse chaque
type avoir son propre threshold data-driven.

### I.2.3 `train` (Lightning + PyTorch)

Voir la **Partie II** ci-dessous pour l'architecture détaillée du modèle.

### I.2.4 `generate`

**Rôle** : à partir d'un BAM (sans kinetics) + un checkpoint, injecter
des kinétiques synthétiques compatibles.

Trois modes auto-détectés :
- **Directory mode** : `kinsim generate <pbsim3_dir> <ckpt> <motifs> <outdir>`
  → pipeline PBSIM3 standard
- **BAM mode** : `kinsim generate <in.bam> <ref.fna> <ckpt> <motifs> <out.bam>`
  → stripped BAM (notre cas pour bc2046)
- **Per-genome mode** : 3-argument fallback

### I.2.5 `verify-generate`

Per-(kmer, meth) comparison entre BAM réel et BAM généré.

---

# Partie II — Architecture du modèle `ConvPredictor`

## II.1 Vue d'ensemble

`ConvPredictor` (`kinsim/models/predictor.py`), ~140K paramètres au total :

1. **Per-base embedding** : 4 bases × 16-dim → `(B, 11, 16)`
2. **Positional embedding** : 11 positions × 16-dim (learnable parameter)
3. **FiLM conditioning** : injection de la méthylation via `(γ, β)`
4. **Conv1D backbone** : 3 layers, kernel=3, BatchNorm + GELU, conv_dim=128
5. **Dual readout** : centre position + global average pool → `(B, 256)`
6. **Output head** : `(μ_ipd, μ_pw, log_σ_ipd, log_σ_pw)` en log1p space

## II.2 Décisions de conception et justifications

### II.2.1 ConvPredictor (140K params) vs MLPPredictor (268M params)

`MLPPredictor` (legacy) utilise une embedding table de 4.2M entrées
(une par 11-mer). Avantage : lookup direct. Inconvénients :
- Impossible de généraliser à un kmer jamais vu pendant le training
- Pas d'apprentissage compositionnel ("G à offset -3 du site actif décale
  IPD de X")
- 268M paramètres dont 99.98% dans la table = capacité gaspillée

`ConvPredictor` apprend des **règles compositionnelles** via :
- Per-base embedding partagé entre les positions (translation equivariance)
- Conv1D pour les patterns locaux et moyenne-portée
- ~140K paramètres → 1900× plus petit

**Critique pour l'imbalance 99/1 unmethylated/methylated** : l'effet de la
méthylation est appris comme une **modulation globale** (via FiLM) plutôt
qu'indépendamment par k-mer — sinon le modèle aurait besoin d'observer
chaque (kmer, scénario) plein de fois.

### II.2.2 Per-base + positional embedding

**Per-base embed (4 × 16)** : chaque base (A, C, G, T) → vecteur 16-dim
appris. Capture les propriétés intrinsèques (taille, hydrogen bonds, etc.).

**Positional embed (11 × 16)** : chaque position dans le 11-mer a sa
propre signature appris. Justification : la position de la base par
rapport au site actif de la polymérase a un effet distinct (la position
-3 contribue à la stiffness, +0 est le site d'incorporation, etc.).

Initialisation `pos_embed ~ N(0, 0.02)` — petit pour ne pas dominer
l'embed at start.

### II.2.3 FiLM conditioning (Perez et al. 2018)

**FiLM** = Feature-wise Linear Modulation. Au lieu de concaténer la
méthylation aux features de base (concat → expansion linéaire des
paramètres), on génère des coefficients `(γ, β)` qui modulent les
features existantes :

```
x_modulated = (1 + γ) · x_base + β
```

`γ, β` sont produits par une projection :
```
meth_full[B, 14, 4] → flatten → meth_proj[Linear(56, 8)] → meth_feat[B, 8]
                              ↓
                    (optional kmer-aware concat)
                              ↓
                  film_in[B, 8 or 24]
                              ↓
                     ┌────────┴────────┐
              film_gamma[B, 16]   film_beta[B, 16]
                     │                 │
                     ↓                 ↓
                broadcast over 11 positions
                            │
                            ↓
                  modulate base_embed[B, 11, 16]
```

**Pourquoi FiLM :**
- Préserve la structure spatiale des embeddings (translation equivariance
  des convs en aval)
- Peu de paramètres ajoutés (~600 pour les deux couches gamma/beta)
- **Init zéro sur film_gamma et film_beta** : `(1+0)·x + 0 = x` → identité
  initiale → start training s'effectue sans interférence de FiLM

**Référence :**
- Perez, Strub, de Vries, Dumoulin, Courville 2018, **"FiLM: Visual
  Reasoning with a General Conditioning Layer"**, AAAI 2018.
  https://arxiv.org/abs/1709.07871

### II.2.4 Kmer-aware FiLM (option `kmer_aware_film=True`)

Sans cette option, `(γ, β)` ne dépendent QUE de la méthylation — la
modulation est identique pour tous les kmers. Mais l'effet cinétique d'une
m6A varie par contexte (motif family).

Avec `kmer_aware_film=True`, on concatène un kmer-summary (mean-pooled
base_embed avant FiLM) à `meth_feat` :

```
film_in = concat(meth_feat[B, 8], kmer_summary[B, 16]) → [B, 24]
```

Le modèle peut alors apprendre des modulations context-dependent. Coût :
+~400 paramètres.

### II.2.5 Conv1D backbone

3 layers (default) :
- Conv1D(in_ch → conv_dim=128, kernel=3, padding=1) — capture les
  patterns locaux (triplets de bases)
- BatchNorm1d — stabilise le training, accélère la convergence
- GELU — activation smooth (différentiable partout, contrairement à ReLU)

**3 layers** → receptive field cumulatif = 1 + 2·1 + 2·1·1 = 5 bases (avec
kernel=3 et padding=1). Mais avec la dual readout (pooling global), tout
le 11-mer contribue effectivement.

**Pourquoi pas de stride / pooling intermédiaire :** on veut préserver la
résolution per-position pour le readout central. Pooling fait à la fin.

### II.2.6 Dual readout (centre + global pool)

Au lieu de juste prendre la position centrale (`x[:, :, 7]`), on
concatène avec le global average pool :

```
center = x[:, :, KMER_PRED_IDX]   # (B, 128)
global_pool = x.mean(dim=2)        # (B, 128)
readout = concat(center, global_pool)  # (B, 256)
```

**Justification :** le centre capture le contexte local au site
d'incorporation ; le global pool capture les effets long-range (e.g. un
G riche en bases AT à distance 5 modifie aussi la cinétique).

### II.2.7 Output head : `(μ_ipd, μ_pw, log_σ_ipd, log_σ_pw)`

Sortie en **log1p space** (les targets le sont aussi via `log_transform`).
Le post-processing à l'inférence applique `inv_log_transform = expm1`
clampé à [0, 255] pour produire des valeurs uint8 compatibles BAM.

**Pourquoi log1p :** distribution PacBio IPD/PW est lourdement
right-skewed (queue méthylation). log1p compresse cette queue tout en
préservant la résolution autour de 0 (baseline).

### II.2.8 Fenêtre asymétrique [-7, +3]

**Constants** (`kinsim/utils/encoding.py`) :
- `K = 11` — taille du 11-mer
- `KMER_LEFT_PAD = 7` — 7 bases en amont de la position de prédiction
- `KMER_RIGHT_PAD = 3` — 3 bases en aval
- `KMER_PRED_IDX = 7` — la position de prédiction est l'index 7 dans le
  11-mer

**Justification biologique :**
1. La polymérase PacBio a lu plus de bases EN AMONT qu'EN AVAL à tout
   instant — l'effet "stiffness" upstream est plus important
2. **TOUTES les signatures cinétiques de méthylation sont AU SITE ou EN
   AVAL** :
   - m6A : signal à +0 et +5
   - m4C : signal à +0
   - m5C : signal à +2 et +6
3. Inspiration directe : `ipdSummary` null model `[-7, +2]` (Feng et al.
   2013)

## II.3 Loss et inférence

### II.3.1 Gaussian NLL (loss standard)

```
L = 0.5 · [ 2 · log σ + (y − μ)² / σ² ]
```

**Problème observé** : modèle peut "tricher" en élargissant σ pour réduire
la pénalité `(y−μ)²/σ²` au lieu d'améliorer μ. Voir Partie III pour le
fix (Beta-NLL).

### II.3.2 Inférence stochastique (`sample()`)

```
log_σ = clamp(model_out[2:], -6, log_sigma_clamp_max)
σ = exp(log_σ)
sample = μ + σ · randn()
output_uint8 = clamp(expm1(sample), 0, 255).round()
```

`log_sigma_clamp_max` est configurable par checkpoint (default 3.0, recommandé
1.5 pour les retrains v13+).

---

# Partie III — Modifications majeures apportées au modèle

## III.1 Biology mask (Décembre 2026 — fix m4C-on-A bug)

**Problème diagnostiqué** : sur le kmer `AAAAAAGATCA` (A central), le
modèle prédit :
- `m6A@+0` : boost 2.49× ✓ (biologiquement valide)
- `m4C@+0` : boost 2.39× ✗ (m4C ne peut PAS exister sur A)
- `m5C@+2` : boost 2.43× ✗ (m5C non plus)

Le modèle a appris le **raccourci** "meth_flag présent à la prediction
position → boost", sans la contrainte base-spécifique.

**Fix architectural** : tampon `_meth_compat[4, M]` construit depuis YAML.
Au forward pass :
```python
compat_at_pos = self._meth_compat[bases]   # (B, 11, 4)
meth_full[:, :11, :] *= compat_at_pos       # zéro les flags impossibles
```

Le modèle ne voit JAMAIS d'inputs biologiquement impossibles → ne peut
pas apprendre dessus.

**YAML-driven** : ajouter un nouveau type (e.g. `m6mA: {modified_base: A,
...}`) étend automatiquement la table. Plusieurs types peuvent cibler la
même base sans s'effacer mutuellement.

**Backward-compat** : registered avec `persistent=False` → ne casse pas
le chargement des anciens checkpoints. Active automatiquement à
l'inférence pour TOUS les checkpoints (fixe le bug même sans retrain).

## III.2 Configurable `log_sigma_clamp_max`

Avant : hardcoded `[-6, 3]` → σ_log peut atteindre e³ ≈ 20.
Après : attribut `log_sigma_clamp_max` configurable par checkpoint via
CLI `--log-sigma-clamp-max`.

Sauvegardé dans `model_config.json`. Backward-compat : default 3.0.

**Recommandation pour retrain v13+** : `--log-sigma-clamp-max 1.5` →
σ_log_max ≈ 4.5 → σ_raw plus serré (CV typique attendu : 10-20% au lieu
de 40%).

## III.3 Paired-positive augmentation

Voir Partie IV ci-dessous.

## III.4 Per-(kmer, category) weighted balancing

Voir Partie IV ci-dessous.

## III.5 Beta-NLL loss

Voir Partie V ci-dessous.

## III.6 Cosine LR schedule + warmup

Voir Partie V ci-dessous.

## III.7 Best `val_mse_mu` checkpoint séparé

Voir Partie V ci-dessous.

---

# Partie IV — Améliorations data pipeline (`kinsim/data/dataset.py`)

## IV.1 Paired-positive augmentation

**Problème** : le modèle apprend "meth_flag présent → IPD boost" mais pas
la spécificité (kmer, meth_type, offset). Ratios uniformes ~2-2.5× quel
que soit le scénario.

**Solution** : pour chaque row non-baseline, **yield aussi un row baseline
réel du même kmer** (random pick parmi la pool baseline). Forces le
modèle à voir le contraste meth/no-meth sur la **même séquence**.

**Pure data augmentation** : pas de mislabel, toutes les données sont
réelles, tous les labels sont les vrais signaux observés. Le contraste
émerge naturellement.

**Décision design** : on a CONSIDÉRÉ puis REJETÉ une variante "fake_neg"
où on injecterait un flag biologiquement impossible (e.g. m5C sur A) avec
target = baseline. Pourquoi rejet :
- C'est un mislabel artificiel ; risque de dampener le vrai signal
- La biology mask architecture (Partie III.1) gère déjà ce cas
  proprement, sans data fakée

**Référence (training par paires de positifs réels)** :
- Bromley et al. 1993, "Signature verification using a siamese time delay
  neural network" — fondateur du paradigme paire
- Chen et al. 2020 (**SimCLR**), "A Simple Framework for Contrastive
  Learning of Visual Representations", ICML 2020.
  https://arxiv.org/abs/2002.05709
- Khosla et al. 2020, "Supervised Contrastive Learning", NeurIPS 2020.
- Kaushik, Hovy, Lipton 2020, "Learning the Difference that Makes a
  Difference with Counterfactually-Augmented Data", ICLR 2020.

## IV.2 Per-(kmer, category) weighted sampling

**Problème** : déséquilibre INTRA-shard. Pour une souche donnée, le motif
méthylé dominant peut avoir 100× plus de rows SLOWED que les motifs
rares. Le modèle apprend bien le dominant, mal les rares.

**Solution** : `ShardedSignalDataset(balance_kmers=True)`.
Pour chaque row :
```
composite_key = kmer_id × N_CATS + category   # N_CATS = 4
weight = 1 / count(composite_key)
```

Puis `np.random.choice(n, size=cap, p=weights)`. Chaque bucket
(kmer × category) reçoit la même attention attendue par epoch,
indépendamment de son volume natif.

**Pourquoi par-(kmer, category) plutôt que par-kmer seul :** un kmer
peut avoir 50 BASELINE rows et 1000 SLOWED. Avec un poids 1/count(kmer
seul), 95% du budget de ce kmer iraient aux SLOWED. Avec par-(kmer,
category), BASELINE et SLOWED reçoivent budget équivalent → le contraste
est mieux appris.

**Référence (class imbalance) :**
- He, Garcia 2009, "Learning from Imbalanced Data", IEEE TKDE — survey
  foundational.
- Cui et al. 2019, "Class-Balanced Loss Based on Effective Number of
  Samples", CVPR.
- Le pattern de WeightedRandomSampler PyTorch ne s'applique pas
  directement aux `IterableDataset` (qu'on a). On implémente la même
  logique via `numpy.random.Generator.choice(p=weights)` au load de
  chaque shard.

---

# Partie V — Améliorations training (`kinsim/train.py`)

## V.1 Beta-NLL loss (Seitzer et al. 2022)

**Problème** : Gaussian NLL standard a un échec connu — le modèle peut
satisfaire la loss en augmentant σ plutôt qu'améliorant μ. Observé chez
nous : σ_log ≈ 0.35 → σ_raw ≈ 20 sur μ=50 (CV ~40%).

**Solution** : Beta-NLL loss :

```
L_β = 0.5 · σ²(.detach())^β · [ 2 · log σ + (y − μ)² / σ² ]
                       ↑
                  re-pondération stop-grad
```

`β=0.5` : recommandé par Seitzer. Stop-grad sur σ² → le modèle ne peut
PAS optimiser le poids en élargissant σ ; il doit améliorer μ pour faire
baisser `(y−μ)²`.

**Référence :**
- Seitzer, Pirinen, Welling 2022, **"On the Pitfalls of Heteroscedastic
  Uncertainty Estimation with Probabilistic Neural Networks"**, ICLR
  2022. https://arxiv.org/abs/2203.09168
- Lakshminarayanan, Pritzel, Blundell 2017, "Simple and Scalable
  Predictive Uncertainty Estimation Using Deep Ensembles", NeurIPS —
  baseline heteroscedastic NN.

CLI : `--loss betanll` (β=0.5), `--loss betanll_0.3`, `--loss betanll_1.0`.

## V.2 Cosine LR schedule + linear warmup

**Problème** : `ReduceLROnPlateau` réagit à `val_loss`, qui est bruité
au début de l'entraînement (dominé par le terme σ de la GNLL). Le LR
peut être réduit trop tôt.

**Solution** : `--lr-schedule cosine`. Schedule :
- **Epochs 0..warmup-1** : linear ramp 0.01·LR → 1.0·LR
- **Epochs warmup..total-1** : cosine decay 1.0·LR → 0.01·LR

Implémenté via `torch.optim.lr_scheduler.LambdaLR`. Découplé de
`val_loss`.

**Référence :**
- Loshchilov, Hutter 2017, "SGDR: Stochastic Gradient Descent with Warm
  Restarts", ICLR — cosine annealing original.
- Goyal et al. 2017, "Accurate, Large Minibatch SGD: Training ImageNet in
  1 Hour", arXiv — warmup linéaire.

## V.3 `best_val_mse_mu` checkpoint séparé

**Problème** : `val_loss` (GNLL) est dominé tôt par σ, donc le "best
val_loss" peut être atteint avant que les μ aient convergé. On a observé
sur v7e que le best checkpoint était à epoch 0, donc μ très peu raffinés.

**Solution** : second `ModelCheckpoint` qui monitor `val_mse_mu` (moyenne
de val_mse_ipd et val_mse_pw). On garde le top-1 sur cette métrique.

Aux extractions ultérieures (`kinsim predict-kmers`, `kinsim generate`),
on peut choisir `best_mu_*.ckpt` plutôt que `ckpt-*-val_loss=*.ckpt` si
la qualité des moyennes prime sur la calibration σ.

---

# Partie VI — Améliorations inference (`kinsim/generate.py` + LUT)

## VI.1 Vectorisation du chemin unmapped

**Problème** : `kinsim generate` traitait ~4 reads/sec → 8h+ pour bc2046
(680k reads). Le GPU était inactif 97% du temps ; le goulot était les
boucles Python per-position.

**Solution** : helper `_process_read_unmapped_vec` — toutes les
opérations per-position remplacées par numpy whole-array :
- Encoding rolling kmer via `np.lib.stride_tricks.sliding_window_view` +
  matmul
- Construction de `meth_full` vectorisée
- `_apply_p_fire_to_mc` vectorisé (Bernoulli matriciel)
- `_rc_kmer_vec` pour les reverse-complement kmers

**Gain** : 268 ms/read → 18-30 ms/read (~10-15× speedup).

## VI.2 Multi-threaded BAM I/O

```python
pysam.AlignmentFile(input_bam, "rb", check_sq=False, threads=4)
pysam.AlignmentFile(output_bam, "wb", header=header_out, threads=4)
```

htslib utilise 4 threads natifs pour BGZF (de)compression en arrière-plan.
Speedup ~2× sur l'I/O.

## VI.3 `tobytes()` au lieu de `.tolist()` pour BAM tags

Avant : 10k Python int objects par tag, 4 tags par read × 1000 reads ≈
40s/batch d'overhead Python.

Après : `bytes` direct (memcpy), ~ms/batch.

## VI.4 Flag `--region` pour array-job parallelism

Permet de lancer N jobs SLURM array, chacun avec `--region chr:A-B`.
Chaque job traite seulement les reads de sa région via `pysam.fetch()`.
Wall time pour bc2046 : 3h → ~20 min avec N=10.

Requires un BAM indexé (`samtools index`).

## VI.5 `--use-lookup` (model distillation)

**Concept** : à l'inférence, le modèle est consulté pour 4²² × N_scenarios
inputs. On peut pré-calculer (μ, σ) pour TOUS ces inputs via
`kinsim predict-kmers` et stocker en .npz (~384 MB).

Pendant `generate`, on consulte le LUT plutôt que d'appeler le modèle :
- Pas de GPU nécessaire (pure numpy)
- 1000× plus rapide sur le forward pass
- Trivialement parallélisable (LUT shared read-only)

**Approximation** : pour les positions où plusieurs méthylations se
chevauchent dans la fenêtre 11-mer (<0.01% sur E. coli/Strepto), le LUT
prend le scénario "dominant". Acceptable.

**Référence :**
- Hinton, Vinyals, Dean 2015, "Distilling the Knowledge in a Neural
  Network", NeurIPS workshop. https://arxiv.org/abs/1503.02531

## VI.6 `KINSIM_USE_REF_CTX` env gate

Par défaut, `generate` route TOUS les reads par le chemin "unmapped"
vectorisé, même pour les BAMs alignés. L'edge-accuracy gain du chemin
"mapped" (avec ref_context pour les bords des reads) est statistiquement
négligeable sur les BAMs typiques (~1% des positions sont à <K bases d'un
read end).

`KINSIM_USE_REF_CTX=1` réactive l'ancien chemin pour les cas où la
précision aux bords compte (e.g. read-end analyses).

---

# Partie VII — Outils nouveaux et utilitaires

## VII.1 `kinsim_baseline per-kmer` + `plot-per-kmer`

Nouveaux modules dans `kinsim_baseline/`.

**`per_kmer.py`** : walk les BAMs du manifest, accumule par kmer :
- `n_total`, `n_above_ipd`, `n_above_pw` (count above threshold)
- `sum_obs`, `sum2_obs` pour les stats μ, σ empiriques
- `hist_ipd[kmer, 64 bins]` : histogramme IPD complet

**`plot_kmer.py`** : 3-panel HTML :
- Scatter `μ_pred` (AI baseline) vs `μ_obs` (empirique)
- Distribution above-rate par kmer
- Top-K kmers detail avec AI Gaussian (bleue) + empirique (rouge
  pointillé) + histogramme observé (barres grises)

**Caveat documenté** : la métrique "above-rate" ne valide PAS la baseline
de l'IA — elle identifie les kmers enrichis aux sites méthylés du corpus.
Pour une vraie validation baseline, il faut conditionner sur `meth_ctx`.

## VII.2 Scripts diagnostiques (`scripts/`)

| Script | Rôle |
|---|---|
| `predict_kmer.py` | Print les prédictions AI per-scenario pour un 11-mer donné. Diagnostique le bug m4C-on-A. |
| `count_modified_positions.py` | Compte BAM-level les bases INSIDE vs OUTSIDE motif sites. Démontre l'imbalance ~0.01% / 99.99% qui motive le baseline kmer-aware. |
| `baseline_threshold_view.py` | Visualise l'émergence du signal méthylation à différents seuils (1.0×, 1.3×, 1.5×, 2.0× × baseline). |
| `strip_kinetics.py` | Fix `__main__` block manquant. |

---

# Partie VIII — Points faibles identifiés du modèle (slides "limitations")

Diagnostiqués empiriquement sur le checkpoint v7e (corpus 52 Strepto +
Vega) :

| Limitation | Preuve empirique | Cause racine | Mitigation |
|---|---|---|---|
| **σ prédit trop large** | σ_log ≈ 0.35 → σ_raw ≈ 20 sur μ=50 (CV ~40%) | GNLL standard permet de "tricher" σ | Beta-NLL + clamp `log_sigma_clamp_max=1.5` |
| **m4C/m5C boost sur A central** | Ratio 2.39× sur `AAAAAAGATCA` (centre=A) | Pas de contrainte base × meth_type au training | Biology mask architectural (Partie III.1) |
| **Ratios uniformes ~2-2.5× tous scénarios** | m6A@+0 = 2.49, m4C@+0 = 2.39, m5C@+2 = 2.43 — quasi identiques | Sous-entraînement + raccourci "meth flag = boost" | Paired augmentation + plus d'époques |
| **Best checkpoint = epoch 0** | val_loss montait après ; modèle pas eu le temps d'apprendre | val_NLL bruité tôt par σ | Cosine LR + best_val_mse_mu checkpoint séparé |

---

# Partie IX — Future work (au-delà du stage)

1. **Per-occupancy `p_fire` curve** — actuellement `p_fire = target_frac ×
   p_efficiency` est linéaire en occupancy. Pour weak-signal types
   (m4C, m5C), stratifier par fraction bin (0-0.3, 0.3-0.6, 0.6-1.0) et
   stocker une curve.

2. **Wider `rev_meth` window** — actuellement [-1, 0, +1] pour les
   neighbours active-site. Pour motifs palindromiques 8+ bp (Type II R-M
   denses), le partner methyl peut être à ±3-5. Layout change + retrain.

3. **Multi-process generate** — split BAM en N shards, run en parallèle.
   Combiné avec `--use-lookup` pour CPU-only, scale au-delà de 1 GPU.

4. **Ensemble** — train N modèles avec seeds différents, average leurs
   prédictions. Réduit la variance.

5. **Active learning** — re-sampler les kmers high-loss en priorité après
   chaque epoch.

6. **Mixed precision bf16** — 2× speedup sur GPUs modernes. Tentative
   ratée sur ce cluster spécifique (instabilité numérique), mais standard
   ailleurs.

7. **Wider supervision** — utiliser modkit/jasmine outputs comme labels
   semi-supervisés pour les positions qu'ipdSummary ne flag pas.

---

# Partie X — Reproduction des résultats

## X.1 Re-training v13 avec toutes les améliorations

Depuis le redesign CLI (décembre 2026), **toutes les améliorations sont ON
par défaut**. Commande minimale :

```bash
kinsim train shards_refined/ checkpoints_v13/ --epochs 50
```

Cela active automatiquement :
- Paired-positive augmentation
- Per-(kmer, category) balanced sampling
- Biology mask architectural
- `log_sigma_clamp_max = 1.5` (σ resserrée)
- Cosine LR schedule + 3 epochs warmup
- Deuxième checkpoint `best_val_mse_mu`

**Avec Beta-NLL en plus** (pénalise σ encore plus fort) :

```bash
kinsim train shards_refined/ checkpoints_v13/ --epochs 50 --loss betanll
```

**Recreate v7e behavior (ablation)** :

```bash
kinsim train shards_refined/ checkpoints_v7e_repro/ \
    --epochs 50 \
    --no-augment --no-balance-kmers --no-biology-mask \
    --log-sigma-clamp-max 3.0 --lr-schedule plateau
```

## X.2 Inférence (biology mask auto-activé)

```bash
kinsim generate stripped.bam ref.fna checkpoint.pt motifs.csv simulated.bam
```

## X.3 LUT mode (CPU-only, no GPU needed)

```bash
kinsim predict-kmers checkpoints_v13/ predict_kmers_v13
kinsim generate stripped.bam ref.fna checkpoint.pt motifs.csv simulated.bam \
    --use-lookup predict_kmers_v13.npz --device cpu
```

## X.4 Validation per-kmer baseline (kmer-aware analysis)

```bash
python -m kinsim_baseline per-kmer \
    predict_kmers_v13.npz manifest.csv per_kmer_out/ \
    --threshold 2.0
python -m kinsim_baseline plot-per-kmer per_kmer_out/
```

---

# Bibliographie consolidée

## Loss / training methodology

1. **Seitzer, M., Pirinen, A., Welling, M.** (2022). On the Pitfalls of
   Heteroscedastic Uncertainty Estimation with Probabilistic Neural
   Networks. *ICLR 2022*. https://arxiv.org/abs/2203.09168

2. **Lakshminarayanan, B., Pritzel, A., Blundell, C.** (2017). Simple and
   Scalable Predictive Uncertainty Estimation Using Deep Ensembles.
   *NeurIPS 2017*.

3. **Loshchilov, I., Hutter, F.** (2017). SGDR: Stochastic Gradient
   Descent with Warm Restarts. *ICLR 2017*.

4. **Goyal, P. et al.** (2017). Accurate, Large Minibatch SGD: Training
   ImageNet in 1 Hour. arXiv:1706.02677.

5. **Hinton, G., Vinyals, O., Dean, J.** (2015). Distilling the Knowledge
   in a Neural Network. *NeurIPS workshop 2015*.
   https://arxiv.org/abs/1503.02531

## Contrastive / paired learning

6. **Bromley, J. et al.** (1993). Signature Verification using a Siamese
   Time Delay Neural Network. *NeurIPS 1993*.

7. **Chen, T., Kornblith, S., Norouzi, M., Hinton, G.** (2020). A Simple
   Framework for Contrastive Learning of Visual Representations (SimCLR).
   *ICML 2020*. https://arxiv.org/abs/2002.05709

8. **Khosla, P. et al.** (2020). Supervised Contrastive Learning.
   *NeurIPS 2020*.

9. **Kaushik, D., Hovy, E., Lipton, Z.** (2020). Learning the Difference
   that Makes a Difference with Counterfactually-Augmented Data.
   *ICLR 2020*.

## Architecture

10. **Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A.**
    (2018). FiLM: Visual Reasoning with a General Conditioning Layer.
    *AAAI 2018*. https://arxiv.org/abs/1709.07871

11. **Avsec, Ž. et al.** (2021). Effective gene expression prediction
    from sequence by integrating long-range interactions (Enformer).
    *Nature Methods 18*. https://www.nature.com/articles/s41592-021-01252-x

## Class imbalance

12. **He, H., Garcia, E. A.** (2009). Learning from Imbalanced Data.
    *IEEE Transactions on Knowledge and Data Engineering 21(9)*.

13. **Cui, Y. et al.** (2019). Class-Balanced Loss Based on Effective
    Number of Samples. *CVPR 2019*.

## PacBio kinetics / méthylation biology

14. **Flusberg, B. et al.** (2010). Direct detection of DNA methylation
    during single-molecule, real-time sequencing. *Nature Methods 7*.
    https://www.nature.com/articles/nmeth.1459

15. **Feng, Z. et al.** (2013). Detecting DNA modifications from SMRT
    sequencing data by modeling sequence context dependence of
    polymerase kinetic rate. *PLoS Computational Biology*. (KineticsTools
    / `ipdSummary`)

16. **Beaulaurier, J. et al.** (2019). Metagenomic binning and
    association of plasmids with bacterial host genomes using DNA
    methylation. *Nature Biotechnology 37*.

17. **Tse, O. Y. O. et al.** (2021). Genome-wide detection of cytosine
    methylation by single molecule real-time sequencing. PNAS.

18. **Tourancheau, A., Mead, E. A., Zhang, X. S., Fang, G.** (2021).
    Discovering multiple types of DNA methylation from bacteria and
    microbiome using nanopore sequencing. *Nature Methods 18*.

## Outils / containers

19. **Li, H. et al.** (2009). The Sequence Alignment/Map format and
    SAMtools. *Bioinformatics 25(16)*. (samtools)

20. **Bonfield, J. K. et al.** (2021). HTSlib: C library for reading/
    writing high-throughput sequencing data. *GigaScience 10(2)*.
    (htslib used through pysam)
